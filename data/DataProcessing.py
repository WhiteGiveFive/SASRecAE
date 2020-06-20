import gzip
from collections import defaultdict
from datetime import datetime


def parse(path):
    """
    This function is from the dataset coding instruction http://jmcauley.ucsd.edu/data/amazon/links.html
    :param path:
    :return:
    """
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0) # create a defaultdict with default value 0, this dictionary is used to count #user
countP = defaultdict(lambda: 0) # create a defaultdict with default value 0, this dictionary is used to count #product
line = 0

dataset_name = 'Beauty'
f = open('reviews_' + dataset_name + '.txt', 'w')

for l in parse('reviews_' + dataset_name + '.json.gz'):
    # This for loop will generate a txt file containing data like below:
    # A39HTATAQ9V7YF 0205616461 5.0 1369699200
    # A3JM6GV9MNOF9X 0558925278 3.0 1355443200
    # A1Z513UWSAAO0F 0558925278 5.0 1404691200
    # A1WMRR494NWEWV 0733001998 4.0 1382572800
    line += 1
    f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    countU[rev] += 1
    countP[asin] += 1
f.close()

usermap = dict()    # This dictionary is used to map revID to an id system from 1 to #users in the dataset.
usernum = 0
itemmap = dict()    # This dictionary is used to map asin to an id system from 1 to #items in the dataset.
itemnum = 0
User = dict()   # The keys of this dictionary are user ids, the values are lists consisting of each user's interaction history with the items.
for l in parse('reviews_' + dataset_name + '.json.gz'):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    if countU[rev] < 5 or countP[asin] < 5:
        continue    # discard users and items with fewer than 5 related actions, continue is to ignore current loop and
                    # continue to next loop

    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid
    User[userid].append([time, itemid])
# sort reviews in User according to time

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])   # referring to sorted/sort key parameter

print usernum, itemnum

# write User dictionary to txt file, in the form of 
# 1   itemid1 in user1 history, the most previous one
# 1   itemid2 in user1 history, the second most previous one
# ...   ...
# 2   itemid1 in user2 history, the most previous one
# 2   itemid2 in user1 history, the second most previous one
# ...   ...
# #users   itemid1 in #users history, the most previous one
# #users   itemid2 in #users history, the second most previous one
# ...   ...

f = open('Beauty.txt', 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))
f.close()
