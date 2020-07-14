import sys
import copy
import random
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

def data_partition(fname):
    """
    This function divide the dataset into training/val/text partitions.
    :param fname: file consisting of the preprocessed data
    :return: a list consisting of 3 dictionaries, each dictionary corresponds one split dataset, and total #item, #user
    """
    usernum = 0 # total user number
    itemnum = 0 # total item number
    User = defaultdict(list)    # dictionary with value of list
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ') #
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)   # The key of the dictionary is int, the user id. The value is a list consisting of item no

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:   # if the interaction of a user with items is less than 3 times
            user_train[user] = User[user]
            user_valid[user] = []   # no valid and text
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2]) # set valid set to the second most recent item in the list
            user_test[user] = []
            user_test[user].append(User[user][-1])  # set test set to the most recent item in the list
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, sess):
    """
    This function sample 100 negative items for each user, together with test item to calculate Hit@10 and NDCG@10.
    :param model: SASrec model
    :param dataset: 3 dictionaries of train, valid and test.
    :param args: containing maxlen
    :param sess:
    :return: Hit@10 and NDCG@10
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)    # maximum sample 100000 users
    else:
        users = xrange(1, usernum + 1)
    text_emb = np.load('data/reviews_emb.npy')
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue  # no train and

        seq = np.zeros([args.maxlen], dtype=np.int32)   # zero array with shape (maxlen,)
        idx = args.maxlen - 1   # 200-1
        seq[idx] = valid[u][0]  # the last element of seq is set to valid set of user u
        idx -= 1    # 199-1
        for i in reversed(train[u]):
            seq[idx] = i    # set the remaining maxlen-1 elements of seq to train[u]
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):    # for each user, randomly sample 100 negative items that are not in training set.
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)  # get the 101 items, 100 negative items and 1 test item

        predictions = -model.predict(sess, [u], [seq], item_idx, text_emb)    # the reason that seq is shifted lies here, we make prediction on test item
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)   # calculate NDCG@10
            HT += 1     # calculate Hit@10
        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()  # flush out the data from the buffer

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(xrange(1, usernum + 1), 10000)
    else:
        users = xrange(1, usernum + 1)
    text_emb = np.load('data/reviews_emb.npy')
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], item_idx, text_emb)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print '.',
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# my code, for showing the heatmap of attention weights from trained model
def show_attention(model, dataset, args, sess, epoch):
    """
    This function collects the attention weights from the last self-attention layer. and save the heatmap
    :param model: trained SASRec model
    :param dataset: 3 dictionaries of train, valid and test.
    :param args: containing maxlen
    :param sess:
    :return: attention weights
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # valid_user = 0.0

    # if usernum>10000:
    #     users = random.sample(xrange(1, usernum + 1), 10000)    # maximum sample 100000 users
    # else:
    #     users = xrange(1, usernum + 1)
    text_emb = np.load('data/reviews_emb.npy')
    u = 11 # random sample one single user, set to user 11 first
    # for u in users:

        # if len(train[u]) < 1 or len(test[u]) < 1: continue  # no train and

    seq = np.zeros([args.maxlen], dtype=np.int32)   # zero array with shape (maxlen,)
    idx = args.maxlen - 1   # 200-1
    seq[idx] = valid[u][0]  # the last element of seq is set to valid set of user u
    idx -= 1    # 199-1
    for i in reversed(train[u]):
        seq[idx] = i    # set the remaining maxlen-1 elements of seq to train[u]
        idx -= 1
        if idx == -1: break
    rated = set(train[u])
    rated.add(0)
    item_idx = [test[u][0]]
    for _ in range(100):    # for each user, randomly sample 100 negative items that are not in training set.
        t = np.random.randint(1, itemnum + 1)
        while t in rated: t = np.random.randint(1, itemnum + 1)
        item_idx.append(t)  # get the 101 items, 100 negative items and 1 test item

    attention = model.att_weight(sess, [u], [seq], item_idx, text_emb)    # the reason that seq is shifted lies here, we make prediction on test item

    # draw heatmap of attention weights and save
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention[0][-12:, :], cmap='viridis')
    fontdict = {'fontsize': 10}

    ax.set_xticks(range(args.maxlen))
    ax.set_yticks(range(args.maxlen))

    ax.set_xticklabels(seq.tolist(), fontdict=fontdict, rotation=90)
    ax.set_yticklabels(seq.tolist(), fontdict=fontdict)

    ax.set_xlabel('Attention weights for user {}'.format(u))

    if not os.path.isdir(args.dataset + '_' + args.train_dir + '/attention'):
        os.makedirs(args.dataset + '_' + args.train_dir + '/attention')
    plt.savefig(os.path.join(args.dataset + '_' + args.train_dir + '/attention', 'AttentionWeights'+str(epoch)))
    print('heatmap on epoch {} is saved'.format(epoch))

    return attention


# my code,
def show_emb(model, dataset, args, sess):
    """
    This function collects the attention weights from the last self-attention layer. and save the heatmap
    :param model: trained SASRec model
    :param dataset: 3 dictionaries of train, valid and test.
    :param args: containing maxlen
    :param sess:
    :return: attention weights
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # valid_user = 0.0

    # if usernum>10000:
    #     users = random.sample(xrange(1, usernum + 1), 10000)    # maximum sample 100000 users
    # else:
    #     users = xrange(1, usernum + 1)
    text_emb = np.load('data/reviews_emb.npy')
    u = 11 # random sample one single user, set to user 11 first
    # for u in users:

        # if len(train[u]) < 1 or len(test[u]) < 1: continue  # no train and

    seq = np.zeros([args.maxlen], dtype=np.int32)   # zero array with shape (maxlen,)
    idx = args.maxlen - 1   # 200-1
    seq[idx] = valid[u][0]  # the last element of seq is set to valid set of user u
    idx -= 1    # 199-1
    for i in reversed(train[u]):
        seq[idx] = i    # set the remaining maxlen-1 elements of seq to train[u]
        idx -= 1
        if idx == -1: break
    rated = set(train[u])
    rated.add(0)
    item_idx = [test[u][0]]
    for _ in range(100):    # for each user, randomly sample 100 negative items that are not in training set.
        t = np.random.randint(1, itemnum + 1)
        while t in rated: t = np.random.randint(1, itemnum + 1)
        item_idx.append(t)  # get the 101 items, 100 negative items and 1 test item

    embeddings = model.emb_table(sess, [u], [seq], item_idx, text_emb)    # the reason that seq is shifted lies here, we make prediction on test item
    return embeddings
