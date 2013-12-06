"""These tests rely on an accessible memcache server"""

import thread
import threading
import time
import multiprocessing

import memcache
from memcachelock import (
    RLock, Lock, ThreadRLock, LOCK_UID_KEY_SUFFIX,
    MemcacheLockReleaseError, MemcacheLockCasError,
    MemcacheLockUidError, MemcacheLockGetsError
)

import unittest


TEST_HOSTS = ['127.0.0.1:11211']
TEST_KEY_1 = 'foo'
TEST_KEY_2 = 'bar'


def _delete_test_keys():
    """Make sure no old keys are lying around"""
    memcache.Client(TEST_HOSTS).delete_multi(
        (TEST_KEY_1, TEST_KEY_2, TEST_KEY_1 + LOCK_UID_KEY_SUFFIX,
         TEST_KEY_2 + LOCK_UID_KEY_SUFFIX)
    )


def setUpModule():
    _delete_test_keys()


def tearDownModule():
    _delete_test_keys()


class TestLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mc1 = memcache.Client(TEST_HOSTS, cache_cas=True)
        cls.mc2 = memcache.Client(TEST_HOSTS, cache_cas=True)
        cls.mc3 = memcache.Client(TEST_HOSTS, cache_cas=True)

    def tearDown(self):
        self.mc1.delete_multi((TEST_KEY_1, TEST_KEY_2))

    def check(self, locked, unlocked):
        for lock in locked:
            self.assertTrue(lock.locked())
        for lock in unlocked:
            self.assertFalse(lock.locked())


class TestBasicLogic(TestLock):
    """Run simple tests.

    Only verifies the API, no race test is done.
    """
    def _test_normal(self, LockType):
        # Two locks sharing the same key, a third for crosstalk checking.
        locka1 = LockType(self.mc1, TEST_KEY_1)
        locka2 = LockType(self.mc2, TEST_KEY_1)
        lockb1 = LockType(self.mc3, TEST_KEY_2)

        self.check([], unlocked=[locka1, locka2, lockb1])

        self.assertTrue(locka1.acquire(False))
        self.check([locka1, locka2], unlocked=[lockb1])

        if LockType is Lock:
            self.assertFalse(locka1.acquire(False))

        self.assertFalse(locka2.acquire(False))
        self.check([locka1, locka2], unlocked=[lockb1])

        self.assertRaises(MemcacheLockReleaseError, locka2.release)
        self.check([locka1, locka2], unlocked=[lockb1])

        self.assertEquals(locka2.getOwnerUid(), locka1.uid)

        locka1.release()
        self.check([], unlocked=[locka1, locka2, lockb1])

        self.assertTrue(locka1.acquire())

        del locka1
        # Lock still held, although owner instance died
        self.assertTrue(locka2.locked())

    def _test_reentrant(self, LockType):
        # Basic RLock-ish behaviour
        lock = LockType(self.mc1, TEST_KEY_1)
        self.assertTrue(lock.acquire(False))
        self.assertTrue(lock.acquire(False))
        lock.release()
        self.assertTrue(lock.locked())
        lock.release()
        self.assertFalse(lock.locked())

    def _test_reentrant_thread(self, LockType):
        """Return whether the lock was acquired inside the thread"""
        # I just need a mutable object. Event happens to have the API I need.
        success = threading.Event()

        def release(lock):
            if lock.acquire(False):
                success.set()

        lock = LockType(self.mc1, TEST_KEY_1)
        lock.acquire()
        release_thread = threading.Thread(target=release, args=(lock, ))
        release_thread.daemon = True
        release_thread.start()
        release_thread.join(1)
        self.assertFalse(release_thread.is_alive())
        return success.is_set()

    def test_normal_lock(self):
        self._test_normal(Lock)

    def test_normal_rlock(self):
        self._test_normal(RLock)

    def test_normal_threadrlock(self):
        self._test_normal(ThreadRLock)

    def test_reentrant_rlock(self):
        self._test_reentrant(RLock)

    def test_reentrant_threadrlock(self):
        self._test_reentrant(ThreadRLock)

    def test_threaded_rlock(self):
        self.assertTrue(self._test_reentrant_thread(RLock))

    def test_threaded_threadrlock(self):
        self.assertFalse(self._test_reentrant_thread(ThreadRLock))


class TestMemcacheNoCas(TestLock):
    def test_nocas(self):
        mc = memcache.Client(TEST_HOSTS)
        self.assertRaises(TypeError, Lock, mc, TEST_KEY_1)


class TestMemcacheGoesAway(TestLock):
    """What happens when memcache is no longer around..."""
    def setUp(self):
        self.mc4 = memcache.Client(['127.0.0.1:11211'], cache_cas=True)

    def _bring_down_mc4(self):
        self.mc4.set_servers(['127.0.0.1:12345'])

    def test_init_failure(self):
        self._bring_down_mc4()
        self.assertRaises(
            MemcacheLockUidError, Lock,
            self.mc4, TEST_KEY_1
        )

    def test_acquire_failure(self):
        lock = Lock(self.mc4, TEST_KEY_1)
        self._bring_down_mc4()
        self.assertRaises(MemcacheLockGetsError, lock.acquire, False)

    def test_release_away_failure(self):
        lock = Lock(self.mc4, TEST_KEY_1)
        self.assertTrue(lock.acquire(False))
        self._bring_down_mc4()
        self.assertRaises(
            MemcacheLockGetsError, lock.release
        )

    def test_release_returned_failure(self):
        lock = Lock(self.mc4, TEST_KEY_1)
        self.assertTrue(lock.acquire(False))
        # simulate the key going away
        # (e.g. server reboot, timeout, eviction, ...)
        self.mc4.delete(TEST_KEY_1)
        self.assertRaises(
            MemcacheLockReleaseError, lock.release
        )


class TestMemcacheDurable(TestLock):
    """What happens when memcache is no longer around (but we don't care)"""
    def setUp(self):
        self.mc4 = memcache.Client(TEST_HOSTS, cache_cas=True)

    def _bring_down_mc4(self):
        self.mc4.set_servers(['127.0.0.1:12345'])

    def _bring_up_mc4(self):
        self.mc4.set_servers(TEST_HOSTS)

    def test_init_failure(self):
        self._bring_down_mc4()
        locka = Lock(self.mc4, TEST_KEY_1, durable=True)
        lockb = Lock(self.mc4, TEST_KEY_1, durable=True)
        self.assertIsInstance(locka.uid, str)  # checking uuid used
        self._bring_up_mc4()
        self.assertTrue(locka.acquire(False))
        self.assertFalse(lockb.acquire(False))
        locka.release()
        self.assertTrue(lockb.acquire(False))
        lockb.release()

    def test_acquire_failure(self):
        locka = Lock(self.mc4, TEST_KEY_1, durable=True)
        self._bring_down_mc4()
        self.assertTrue(locka.acquire(False))
        self._bring_up_mc4()
        locka.release()

    def test_release_away_failure(self):
        lock = Lock(self.mc4, TEST_KEY_1, durable=True)
        self.assertTrue(lock.acquire(False))
        self._bring_down_mc4()
        lock.release()

    def test_release_returned_failure(self):
        lock = Lock(self.mc4, TEST_KEY_1, durable=True)
        self.assertTrue(lock.acquire(False))
        # simulate the key going away
        # (e.g. server reboot, timeout, eviction, ...)
        self.mc4.delete(TEST_KEY_1)
        lock.release()


### multiprocessing helper
def locker((LockType, key, sleep_time)):
    lock = LockType(memcache.Client(TEST_HOSTS, cache_cas=True), key)
    lock.acquire()
    if sleep_time:
        time.sleep(sleep_time)
    lock.release()
    return None


class TestTimeout(TestLock):
    def _test_timeout(self, LockType):
        timeout = 1
        locka = LockType(self.mc1, TEST_KEY_1, timeout=timeout)
        lockb = LockType(self.mc2, TEST_KEY_1)
        start = time.time()
        self.assertTrue(locka.acquire(False))
        self.assertTrue(lockb.acquire())
        interval = time.time() - start
        self.assertGreater(interval, timeout - 1)
        self.assertLess(interval, timeout + 1)
        lockb.release()

    def test_timeout_lock(self):
        self._test_timeout(Lock)

    def test_timeout_rlock(self):
        self._test_timeout(RLock)

    def test_timeout_threadrlock(self):
        self._test_timeout(ThreadRLock)


class TestSwarm(TestLock):
    SWARM_SIZE = 30

    def _test_deadlock(self, LockType):
        SLEEP_TIME = 0.001

        pool = multiprocessing.Pool(processes=self.SWARM_SIZE)
        start = time.time()
        list(pool.imap_unordered(
            locker,
            [
                (LockType, TEST_KEY_1, SLEEP_TIME)
                for _ in xrange(self.SWARM_SIZE)
            ]
        ))  # list forces us to get results
        interval = time.time() - start

        self.assertGreater(interval, SLEEP_TIME * self.SWARM_SIZE)
        pool.close()
        pool.join()

    def test_deadlock_lock(self):
        self._test_deadlock(Lock)

    def test_deadlock_rlock(self):
        self._test_deadlock(RLock)

    def test_deadlock_threadrlock(self):
        self._test_deadlock(ThreadRLock)


if __name__ == '__main__':
    unittest.main()
