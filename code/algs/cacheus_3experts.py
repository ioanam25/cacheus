import random

from .lib.dequedict import DequeDict
from .lib.heapdict import HeapDict
from .lib.pollutionator import Pollutionator
from .lib.visualizinator import Visualizinator
from .lib.optional_args import process_kwargs
from .lib.cacheop import CacheOp
import numpy as np


class Cacheus:
    # Entry to track the page information
    class Cacheus_Entry:
        def __init__(self, oblock, freq=1, time=0, is_new=True):
            self.oblock = oblock
            self.freq = freq
            self.time = time
            self.evicted_time = None
            self.is_new = is_new

        # Return min heap (frequency)
        def __lt__(self, other):
            if self.freq == other.freq:
                return self.time > other.time
            return self.freq < other.freq

        # Useful for debugging
        def __repr__(self):
            return "(o={}, f={}, t={})".format(self.oblock, self.freq,
                                               self.time)

    class Cacheus_Learning_Rate:
        def __init__(self, period_length, **kwargs):
            self.learning_rate = 0.1

            process_kwargs(self, kwargs, acceptable_kws=['learning_rate'])

            self.learning_rate_reset = min(max(self.learning_rate, 0.001), 1)
            self.learning_rate_curr = self.learning_rate
            self.learning_rate_prev = 0.0

            self.period_len = period_length

            self.hitrate = 0
            self.hitrate_prev = 0.0
            self.hitrate_diff_prev = 0.0
            self.hitrate_zero_count = 0
            self.hitrate_nega_count = 0

        # Used to use the learning_rate value to multiply without
        # having to use self.learning_rate.learning_rate, which can
        # impact readability
        def __mul__(self, other):
            return self.learning_rate * other

        # Update the adaptive learning rate when we've reached the end of a period
        # ALGORITHM 3 pseudocode
        def update(self, time):
            if time % self.period_len == 0:
                hitrate_curr = round(self.hitrate / self.period_len, 3)
                hitrate_diff = round(hitrate_curr - self.hitrate_prev, 3)

                #delta_LR = lr_{t-i} - lr{t-2i}
                delta_LR = round(self.learning_rate_curr, 3) - round(
                    self.learning_rate_prev, 3)
                delta = self.getSign(delta_LR, hitrate_diff)

                if delta != 0:
                    self.learning_rate = max(
                        self.learning_rate + delta * (self.learning_rate * delta_LR), 0.001)
                    self.hitrate_nega_count = 0
                    self.hitrate_zero_count = 0

                elif delta == 0 and hitrate_diff <= 0:
                    if (hitrate_curr <= 0 and hitrate_diff == 0):
                        self.hitrate_zero_count += 1
                    if hitrate_diff < 0:
                        self.hitrate_nega_count += 1
                        self.hitrate_zero_count += 1
                    if self.hitrate_zero_count >= 10:
                        self.learning_rate = self.learning_rate_reset
                        self.hitrate_zero_count = 0
                    elif hitrate_diff < 0:
                        if self.hitrate_nega_count >= 10:
                            self.learning_rate = self.learning_rate_reset
                            self.hitrate_nega_count = 0
                        else:
                            self.updateInRandomDirection()

                self.learning_rate_prev = self.learning_rate_curr
                self.learning_rate_curr = self.learning_rate
                self.hitrate_prev = hitrate_curr
                self.hitrate_diff_prev = hitrate_diff
                self.hitrate = 0


        # Update the learning rate according to the change in learning_rate and hitrate
        # learning_rate_diff: lr_{t-i} - lr{t-2i}
        # hit_rate_diff: HR_{t} - HR_{t-i}
        def getSign(self, learning_rate_diff, hitrate_diff):
            delta = learning_rate_diff * hitrate_diff
            # Get delta = 1 if learning_rate_diff and hitrate_diff are both positive or negative
            # Get delta =-1 if learning_rate_diff and hitrate_diff have different signs
            # Get delta = 0 if either learning_rate_diff or hitrate_diff == 0
            sign = 1 if delta > 0 else -1
            if delta == 0:
                sign = 0
            return sign

        # Update the learning rate in a random direction
        def updateInRandomDirection(self):
            # choose randomly a learning rate in [10^{-3}, 1)
            self.learning_rate = np.random.uniform(0.001, 1)

    # ALGORITHM 1 caching
    def __init__(self, cache_size, window_size, **kwargs):
        # Randomness and Time
        np.random.seed(123)
        self.time = 0

        # Cache
        self.cache_size = cache_size
        self.s = DequeDict()
        self.q = DequeDict()
        self.r = DequeDict()

        # lfu heap
        self.lfu = HeapDict()

        # Histories
        self.history_size = cache_size // 3
        self.lru_hist = DequeDict()
        self.lfu_hist = DequeDict()
        self.random_hist = DequeDict()

        # Learning Rate
        self.learning_rate = self.Cacheus_Learning_Rate(
            cache_size, **kwargs)

        process_kwargs(self,
                       kwargs,
                       acceptable_kws=['initial_weight', 'history_size'])

        # Decision Weights
        # TODO make 3
        self.W = np.array([0.4, 0.4, 0.2], dtype=np.float32)

        # Variables TODO see if we need them
        hits_ratio = 0.01
        self.q_limit = max(1, int((hits_ratio * self.cache_size) + 0.5))
        self.s_limit = self.cache_size - self.q_limit - 0.1 * self.cache_size
        self.r_limit = 0.1 * cache_size
        self.q_size = 0
        self.s_size = 0
        self.r_size = 0

        # Visualize
        self.visual = Visualizinator(
            labels=['W_lru', 'W_lfu', 'W_random', 'hit-rate', 'q_size'],
            windowed_labels=['hit-rate'],
            window_size=window_size,
            **kwargs)
        # Pollution
        self.pollution = Pollutionator(cache_size, **kwargs)

    # check if the page requested is in cache
    def __contains__(self, oblock):
        return (oblock in self.s or oblock in self.q or oblock in self.r)

    def cacheFull(self):
        return len(self.s) + len(self.q) + len(self.r) == self.cache_size

    # Hit in MRU portion of the cache
    def hitinS(self, oblock):
        x = self.s[oblock]
        x.time = self.time
        self.s[oblock] = x
        x.freq += 1
        self.lfu[oblock] = x

    # Hit in LRU portion of the cache
    def hitinQ(self, oblock):
        x = self.q[oblock]
        x.time = self.time

        x.freq += 1
        self.lfu[oblock] = x

        del self.q[x.oblock]
        self.q_size -= 1

        if self.s_size >= self.s_limit:
            y = self.s.popFirst()
            self.s_size -= 1
            self.q[y.oblock] = y
            self.q_size += 1

        self.s[x.oblock] = x
        self.s_size += 1

    # Hit in randomly added part of cache
    def hitinR(self, oblock):
        x = self.r[oblock]
        x.time = self.time
        self.s[oblock] = x
        x.freq += 1
        self.lfu[oblock] = x

    # Add Entry to S with given frequency
    def addToCacheLocation(self, location, oblock, freq, isNew=True):
        x = self.Cacheus_Entry(oblock, freq, self.time, isNew)
        if location == "s":
            self.s[oblock] = x
            self.s_size += 1
        elif location == "q":
            self.q[oblock] = x
            self.q_size += 1
        else:
            self.r[oblock] = x
            self.r_size += 1
        self.lfu[oblock] = x

    # Add Entry to history dictated by policy
    # policy: 0, Add Entry to LRU History
    #         1, Add Entry to LFU History
    #        -1, Do not add Entry to any History
    def addToHistory(self, x, policy):
        # Use reference to policy_history to reduce redundant code
        policy_history = None
        if policy == 0:
            policy_history = self.lru_hist
        elif policy == 1:
            policy_history = self.lfu_hist
        elif policy == -1:
            policy_history = self.random_hist

        # Evict from history is it is full
        if len(policy_history) == self.history_size:
            evicted = self.getLRU(policy_history)
            del policy_history[evicted.oblock]
        policy_history[x.oblock] = x

    # Get the LRU item in the given DequeDict
    def getLRU(self, dequeDict):
        return dequeDict.first()

    def getRandom(self, dequeDict):
        random_index = random.randint(0, len(dequeDict))
        for i, elem in enumerate(dequeDict):
            if i == random_index:
                return elem

    # Get the LFU min item in the LFU (HeapDict)
    # NOTE: does *NOT* remove the LFU Entry from LFU
    def getHeapMin(self):
        return self.lfu.min()

    # Get the random eviction choice based on current weights
    def getChoice(self):
        p = np.random.rand()
        if p < self.W[0]:
            return 0
        elif self.W[0] <= p <= self.W[0] + self.W[1]:
            return 1
        else:
            return -1

    # Evict an entry
    def evict(self):
        lru = self.getLRU(self.q)
        lfu = self.getHeapMin()
        rand = self.getRandom(self.q)

        evicted = lru
        policy = self.getChoice()

        # Since we're using Entry references, we use is to check
        # that the LRU and LFU Entries are the same Entry
        if lru is lfu:
            evicted, policy = lru, -1
        elif policy == 0: # LRU
            evicted = lru
            del self.q[evicted.oblock]
            self.q_size -= 1
        elif policy == 1: # LFU
            evicted = lfu
            if evicted.oblock in self.s:
                del self.s[evicted.oblock]
                self.s_size -= 1
            elif evicted.oblock in self.q:
                del self.q[evicted.oblock]
                self.q_size -= 1
        elif policy == -1:
            evicted = rand
            del self.r[evicted.oblock]
            self.q_size -= 1

        del self.lfu[evicted.oblock]
        evicted.evicted_time = self.time
        self.pollution.remove(evicted.oblock)

        self.addToHistory(evicted, policy)

        return evicted.oblock, policy

    # ALGORITHM 2
    # Adjust the weights based on the given rewards for LRU and LFU
    def adjustWeights(self, rewardLRU, rewardLFU, rewardRandom):
        reward = np.array([rewardLRU, rewardLFU], dtype=np.float32)
        self.W = self.W * np.exp(self.learning_rate * reward)
        self.W = self.W / np.sum(self.W)

        if self.W[0] >= 0.99:
            self.W = np.array([0.98, 0.01, 0.01], dtype=np.float32)
        elif self.W[1] >= 0.99:
            self.W = np.array([0.01, 0.98, 0.01], dtype=np.float32)
        elif self.W[2] >= 0.99:
            self.W = np.array([0.01, 0.01, 0.98], dtype=np.float32)

    # Update cache data structure
    def hitinLRUHist(self, oblock):
        evicted = None
        entry = self.lru_hist[oblock]
        entry.freq = entry.freq + 1
        del self.lru_hist[oblock]
        self.adjustWeights(-0.98, -0.01, -0.01)
        if (self.s_size + self.q_size) >= self.cache_size:
            evicted, policy = self.evict()
        self.addToCacheLocation(entry.oblock, "s", entry.freq, isNew=False)
        self.limitStack()
        return evicted

    def hitinLFUHist(self, oblock):
        evicted = None
        entry = self.lfu_hist[oblock]
        entry.freq = entry.freq + 1
        del self.lfu_hist[oblock]
        self.adjustWeights(-0.01, -0.98, -0.01)

        if (self.s_size + self.q_size) >= self.cache_size:
            evicted, policy = self.evict()

        self.addToCacheLocation(entry.oblock, "q", entry.freq, isNew=False)
        self.limitStack()
        return evicted

    def hitinRandHist(self, oblock):
        evicted = None
        entry = self.random_hist[oblock]
        entry.freq = entry.freq + 1
        del self.random_hist[oblock]
        self.adjustWeights(-0.01, -0.01, -0.98)
        if (self.s_size + self.q_size) >= self.cache_size:
            evicted, policy = self.evict()
        self.addToCacheLocation(entry.oblock, "r", entry.freq, isNew=False)
        self.limitStack()
        return evicted

    def limitStack(self):
        while self.s_size >= self.s_limit:
            #mark demoted
            demoted = self.s.popFirst()
            self.s_size -= 1

            # Moved from s to q
            self.q[demoted.oblock] = demoted
            self.q_size += 1

    # Cache Miss
    def miss(self, oblock):
        evicted = None
        freq = 1
        if (self.s_size + self.q_size + self.r_size) >= self.cache_size:
            evicted, policy = self.evict()
            self.addToCacheLocation(oblock, "q", freq, isNew=True)
            self.limitStack()
        else:
            if self.s_size < self.s_limit and self.q_size == 0:
                self.addToCacheLocation(oblock, "s", freq, isNew=False)
            elif self.q_size < self.q_limit:
                self.addToCacheLocation(oblock, "q", freq, isNew=False)
            elif self.r_size < self.r_limit:
                self.addToCacheLocation(oblock, "r", freq, isNew=False)

        return evicted


    # Process and access request for the given oblock
    def request(self, oblock, ts):
        miss = False
        evicted = None

        self.time += 1

        self.visual.add({
            'W_lru': (self.time, self.W[0], ts),
            'W_lfu': (self.time, self.W[1], ts),
            'W_rand': (self.time, self.W[2], ts),
            'q_size': (self.time, self.q_size, ts)
        })

        self.learning_rate.update(self.time)

        if oblock in self.s:
            self.hitinS(oblock)
        elif oblock in self.q:
            self.hitinQ(oblock)
        elif oblock in self.r:
            self.hitinR(oblock)
        elif oblock in self.lru_hist:
            miss = True
            evicted = self.hitinLRUHist(oblock)
        elif oblock in self.lfu_hist:
            miss = True
            evicted = self.hitinLFUHist(oblock)
        elif oblock in self.random_hist:
            miss = True
            evicted = self.hitinRandHist(oblock)
        else:
            miss = True
            evicted = self.miss(oblock)

        # Windowed
        self.visual.addWindow({'hit-rate': 0 if miss else 1}, self.time, ts)

        # Learning Rate
        if not miss:
            self.learning_rate.hitrate += 1

        # Pollution
        if miss:
            self.pollution.incrementUniqueCount()
        self.pollution.setUnique(oblock)
        if self.time % self.cache_size == 0:
            self.pollution.update(self.time)

        op = CacheOp.INSERT if miss else CacheOp.HIT

        return op, evicted

    def getQsize(self):
        x, y = zip(*self.visual.get('q_size'))
        return y
