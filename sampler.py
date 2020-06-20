import numpy as np
from multiprocessing import Process, Queue  # different tasks running in parallel, speeding up the program


def random_neq(l, r, s):
    """
    return a random int in (l,r) while excluding int in s
    :param l: lower bound of random scope
    :param r: upper bound of random scope
    :param s: a set of the excluding scope
    :return: a random int not in s
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    """
    :param user_train:
    :param usernum:
    :param itemnum:
    :param batch_size: the number of sample users in one batch
    :param maxlen:
    :param result_queue: a multiprocessing Queue object
    :param SEED:
    :return: put an one-batch number of samples into the queue object
    """
    def sample():
        """
        :return: a tuple consisting of three np.arrays with same shape of (max-length,) and a random user id
        seq: consisting of user's most recent max-length interaction with items
        pos: consisting of next item to each corresponding item in seq, a right shilfted version of seq
        neg: consisting of neg example to pos
        user: a random user id
        """
        # padding item's id is 0.
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)  # excluding users with input sequence of length less than 1

        seq = np.zeros([maxlen], dtype=np.int32)    # consisting of user's most recent max-length interaction with items
        pos = np.zeros([maxlen], dtype=np.int32)    # consisting of next item to each corresponding item in seq, a right shilfted version of seq
        neg = np.zeros([maxlen], dtype=np.int32)    # consisting of neg example to pos.
        nxt = user_train[user][-1]
        idx = maxlen - 1    # index for seq, pos, neg

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):   # using reverse because we want retrieve the most recent max-length items in users' historical sequence.
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  # generating one negative item example in neg[idx]
            nxt = i # right shift version of seq for pos
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []  # consisting of tuples from sample, with a length of batch size.
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))   # unzip the list of tuple, one_batch, return a list consisting of unzipped tuple [(user1, user2, ...,), (seq1, seq2,...), ...]


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)   # an Queue object in multiprocessing, used to store data from different process
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()  # get one element from the queue

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
