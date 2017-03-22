import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist

from word2vec_pipeline.utils.data_utils import load_w2vec
from word2vec_pipeline.utils.db_utils import item_iterator
from word2vec_pipeline import simple_config
config = simple_config.load()
target_column = config['target_column']

W = load_w2vec()
n_vocab,n_dim = W.syn0.shape
word2index = dict(zip(W.index2word, range(n_vocab)))

batch_size = 64

##################################################################

# Tensorflow model here

import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[batch_size, n_vocab])
word2vec_layer = tf.constant(W.syn0,shape=(n_vocab,n_dim))
alpha = tf.Variable(tf.ones([n_vocab]))

Y = tf.matmul(X*alpha, word2vec_layer)
Y = tf.nn.l2_normalize(Y, dim=1)
dist = tf.matmul(Y, tf.transpose(Y))
loss = (tf.reduce_sum(dist) - batch_size) / (batch_size**2-batch_size)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#loss = weight_sum_model(batch_size=batch_size)

##################################################################


V = []
for item in item_iterator():
    tokens = item[target_column].split()
    row = np.zeros(n_vocab)
    for w in tokens:
        if w not in word2index:
            continue
        row[word2index[w]] += 1
    V.append(row)

V = np.array(V)

n_samples = V.shape[0]
import random

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    random.shuffle(V)

    k = 0
    while True:


        epoch_loss = []

        for i in range(n_samples//batch_size - 1):
            v_batch = V[i*batch_size:(i+1)*batch_size]

            funcs = [optimizer, loss]
            _, ls = sess.run(funcs, feed_dict={X:v_batch})
            epoch_loss.append(ls)
            

        funcs = [alpha,]
        a, = sess.run(funcs, feed_dict={X:v_batch})

        print k,np.mean(epoch_loss), a.max(), a.min()
        
        if k%10==0:
            df = pd.DataFrame(index=W.index2word)
            df["alpha"] = a
            df.to_csv("solved_weights.csv")
            print df
            
        k+=1

        
#####################################################################


batch_size = len(V)


def objective_func(alpha):
    X = normalize((V*alpha).dot(W.syn0))
    dist = pdist(X, metric='cosine')
    loss = dist.sum()
    loss /= (batch_size*(batch_size-1))/2.0
    print loss,alpha.max(), alpha.min()

    return loss

from scipy.optimize import minimize, fmin

alpha = np.ones(n_vocab)

# Starts off at 31.19, 11.28
print minimize(objective_func, alpha)#, method='Nelder-Mead')


