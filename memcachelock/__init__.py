import thread
import threading
import time
import logging
import uuid
import functools


LOCK_UID_KEY_SUFFIX = '_uid'


class MemcacheLockError(Exception):
    """
    Unexpected memcached reaction.
    Either memcached is misbehaving (refusing to hold a value...) or a
    competitor could acquire the lock (because of software error, or memcached
    evicted data we needed...).
    """
    pass


class MemcacheLockCasError(MemcacheLockError):
    pass


class MemcacheLockGetsError(MemcacheLockError):
    pass


class MemcacheLockReleaseError(MemcacheLockError):
    pass


class MemcacheLockUidError(MemcacheLockError):
    pass


def _swallow_lockerrors(alternative_function):
    """Utility decorator so that we continue 'locking' when memcache absent"""
    def func_wrapper(func):
        @functools.wraps(func)
        def final_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except MemcacheLockError:
                if self.logger:
                    self.logger.warning('Memcache lock failure', exc_info=True)
                if self.durable:
                    return alternative_function()
                else:
                    raise
        return final_wrapper
    return func_wrapper


class RLock(object):
    """
    Attempt at using memcached as a lock server, using gets/cas command pair.
    Inspired by unimr.memcachedlock .
    There is no queue management: lock will not be granted in the order it was
    requested, but to the requester reaching memcache server at the right time.
    See thread.LockType documentation for API.

    Note: RLock ignores local threads. So it is not a drop-in replacement for
    python's RLock. See class ThreadRLock.

    How to break things:
    - create a lock instance, then 2**64 others (trash them along the way, you
      don't need to keep them). Next one will think it is the first instance.
    - restart memcache server while one client has taken a lock (unles your
      memcache server restarts from persistent data properly written to disk).
    - have memcache server prune entries
    """
    reentrant = True

    def __init__(self, client, key, interval=0.05, uid=None, timeout=0,
                 durable=False, log=False):
        """
        client (memcache.Client)
            Memcache connection. Must support cas
            (by default, python-memcached DOES NOT unless you set cache_cas)
        key (str)
            Unique identifier for protected resource, common to all locks
            protecting this resource.
            Can be any reasonable string for use as a memcache key.
            Must not end with LOCK_UID_KEY_SUFFIX.
        interval (int/float, 0.05)
            Period between consecutive lock taking attemps in blocking mode.
        uid (any picklable object, None)
            Unique lock instance identifier for given key.
            If None, a new uid will be generated for each new instance.
            Allows overriding default uid allocation. Can also be handy to
            recover lock when previous instance with given uid died with lock
            acquired.
            WARNING: You must be very sure of what you do before fiddling with
            this parameter. Especially, don't mix up auto-allocated uid and
            provided uid on the same key. You have been warned.
        timeout (int)
            how long to set locks for
        durable (bool)
            don't throw Exceptions if memcache or our locks go away;
            pretend you can acquire locks if memcache is not responsive.
            This is probably useful with timeout.
        log (bool)
            send warnings to a logger (on losing memcache). Again, useful with
            timeout/durable.
        """
        if getattr(client, 'cas', None) is None or getattr(client, 'gets',
                None) is None:
            raise TypeError('Client does not implement "gets" and/or "cas" '
                'methods.')

        if hasattr(client, 'cache_cas') and not client.cache_cas:
            raise TypeError('cache_cas is disabled; cas === set, so no locks')

        if key.endswith(LOCK_UID_KEY_SUFFIX):
            raise ValueError('Key conflicts with internal lock storage key '
                '(ends with ' + LOCK_UID_KEY_SUFFIX + ')')

        self.memcache = client
        # Compute hash once only. Also used to keep lock uid close to the
        # value it manages.
        self.timeout = timeout
        key_hash = hash(key)
        self.key = (key_hash, key)
        self.interval = interval
        self.logger = logging.getLogger(__name__) if log else None
        self.durable = durable

        self.uid = uid
        if not self.uid:
            self.uid = self._find_uid((key_hash, key + LOCK_UID_KEY_SUFFIX))

    @_swallow_lockerrors(lambda: uuid.uuid3(uuid.NAMESPACE_DNS, __name__).bytes)
    def _find_uid(self, uid_key):
        self.memcache.check_key(uid_key[1])
        if self.memcache.gets(uid_key) is None:
            # Nobody has used this lock yet (or it was lost in a server
            # restart). Init to 0. Don't care if it fails, we just need a
            # value to be set.
            self.memcache.cas(uid_key, 0)
        res = self.memcache.incr(uid_key)
        if res is None:
            raise MemcacheLockUidError('incr failed to give number')
        return res

    def __repr__(self):
        return '<%s(key=%r, interval=%r, uid=%r, timeout=%s) at 0x%x>' % (
            self.__class__.__name__,
            self.key[1],
            self.interval,
            self.uid,
            self.timeout,
            id(self),
        )

    @_swallow_lockerrors(lambda: True)
    def acquire(self, blocking=True):
        while True:
            owner, count = self.__get()
            if owner == self.uid:
                # I have the lock already.
                assert count
                if self.reentrant:
                    self.__set(count + 1)
                    return True
            elif owner is None:
                # Nobody had it on __get call, try to acquire it.
                try:
                    self.__set(1)
                except MemcacheLockCasError:
                    # Someting else was faster.
                    pass
                else:
                    # I got the lock.
                    return True
            # I don't have the lock.
            if not blocking:
                break
            time.sleep(self.interval)
        return False

    @_swallow_lockerrors(lambda: None)
    def release(self):
        owner, count = self.__get()
        if owner != self.uid:
            raise MemcacheLockReleaseError(
                '%s: should be owned by me (%s), but owned by %s'
                % (self.key[1], self.uid, owner)
            )
        assert count > 0
        self.__set(count - 1)

    def locked(self, by_self=False):
        """
        by_self (bool, False)
            If True, returns whether this instance holds this lock.
        """
        owner_uid = self.__get()[0]
        return by_self and owner_uid == self.uid or owner_uid is not None

    def getOwnerUid(self):
        """
        Return lock owner's uid.
        Purely informative. Chances are this will not be true anymore by the
        time caller gets this value. Can be handy to recover a lock (see
        constructor's "uid" parameter - and associated warning).
        """
        return self.__get()[0]

    __enter__ = acquire

    def __exit__(self, t, v, tb):
        self.release()

    # BBB
    acquire_lock = acquire
    locked_lock = locked
    release_lock = release

    def __get(self):
        value = self.memcache.gets(self.key)
        if value is None:
            # We don't care if this call fails, we just want to initialise
            # the value.
            self.memcache.add(self.key, (None, 0))
            value = self.memcache.gets(self.key)
            if value is None:
                raise MemcacheLockGetsError('Memcache not storing anything')
        return value

    def __set(self, count):
        cas_result = self.memcache.cas(
            self.key, (count and self.uid or None, count), self.timeout
        )
        if cas_result:
            self.last_set_time = time.time()
        else:
            raise MemcacheLockCasError('Lock stolen')

class Lock(RLock):
    reentrant = False

class ThreadRLock(object):
    """
    Thread-aware RLock.

    Combines a regular RLock with a RLock, so it can be used in a
    multithreaded app like an RLock, in addition to RLock behaviour.
    """
    def __init__(self, *args, **kw):
        # Local RLock-ing
        self._rlock = threading.RLock()
        # Remote RLock-ing
        self._memcachelock = RLock(*args, **kw)

    def acquire(self, blocking=True):
        if self._rlock.acquire(blocking):
            return self._memcachelock.acquire(blocking)
        return False

    def release(self):
        # This is sufficient, as:
        # - if we don't own rlock, it will raise (we won't release memcache)
        # - if memcache release raises, there is no way to recover (we thought
        #   we were owning the lock)
        self._rlock.release()
        self._memcachelock.release()

    __enter__ = acquire

    def __exit__(self, t, v, tb):
        self.release()

    def locked(self):
        if self._rlock.acquire(False):
            try:
                return self._memcachelock.locked()
            finally:
                self._rlock.release()
        return False

    @property
    def uid(self):
        return self._memcachelock.uid

    def getOwnerUid(self):
        return self._memcachelock.getOwnerUid()

    # BBB
    acquire_lock = acquire
    locked_lock = locked
    release_lock = release

# BBB
MemcacheLock = Lock
MemcacheRLock = RLock
ThreadMemcacheRLock = ThreadRLock
