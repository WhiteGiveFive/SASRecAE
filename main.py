import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)  # split the dataset
[user_train, user_valid, user_test, usernum, itemnum] = dataset  # deploy the dataset
# print 'itemnum: %d' % (itemnum)   # my code
num_batch = len(user_train) / args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print 'average sequence length: %.2f' % (cc / len(user_train))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()   # configure the session
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU, increase GPU utilization
config.allow_soft_placement = True  # allow TensorFlow to automatically choose an existing and supported device to run the operations
sess = tf.Session(config=config)    # Driver for Graph execution

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
text_emb = np.load('data/reviews_emb.npy')  # my code, loading the text embedding of item
print('The loaded embedding type is {}'.format(text_emb.dtype)) # my code, show loaded embedding's data type
# text_emb = np.random.rand(itemnum+1, 300)   # my code, for text embeddings
# text_emb = tf.random.uniform(shape=[itemnum+1, 300], minval=0, maxval=1, dtype=tf.float32, seed=10)  # my code
model = Model(usernum, itemnum, args)
# sess.run(tf.global_variables_initializer)   # my code
sess.run(tf.initialize_all_variables())

T = 0.0
t0 = time.time()

try:
    for epoch in range(1, args.num_epochs + 1):

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # get the tuples of user id, seq, pos and neg. e.g. user id tuple = (user1, user2, ...,), seq = (list of user1 seq, list of user2 seq,...)
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.text_emb: text_emb, model.is_training: True})

        if epoch % 1 == 0:
            t1 = time.time() - t0
            T += t1
            print 'attention weights'
            attention = show_attention(model, dataset, args, sess, epoch)   # my code, for show attention
            if not os.path.isdir(args.dataset + '_' + args.train_dir + '/attention'):
                os.makedirs(args.dataset + '_' + args.train_dir + '/attention')
            np.save(os.path.join(args.dataset + '_' + args.train_dir + '/attention', 'attention'+str(epoch)+'.npy'), attention)
            print('extracting embeddings')
            item_emb = show_emb(model, dataset, args, sess) # my code, for extracting and saving item embeddings
            if not os.path.isdir(args.dataset + '_' + args.train_dir + '/itemb'):
                os.makedirs(args.dataset + '_' + args.train_dir + '/itemb')
            np.save(os.path.join(args.dataset + '_' + args.train_dir + '/itemb', 'item_emb_'+str(epoch)+'.npy'), item_emb)
            print('item embedding on epoch {} is saved'.format(epoch))
            print 'Evaluating',
            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print ''
            print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])

            f.write(str(t_valid) + ' ' + str(t_test) + ' ' + str(T) + '\n')
            f.flush()
            t0 = time.time()
except:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
