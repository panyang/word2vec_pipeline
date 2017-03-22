import numpy as np
from tqdm import tqdm

from word2vec_pipeline.utils.data_utils import load_w2vec
from word2vec_pipeline.utils.db_utils import item_iterator
from word2vec_pipeline import simple_config
config = simple_config.load()
target_column = config['target_column']

W = load_w2vec()
n_vocab = len(W.index2word)
word2index = dict(zip(W.index2word, range(n_vocab)))

V = []
for item in tqdm(item_iterator()):
    tokens = item[target_column].split()
    row = np.zeros(n_vocab)
    for w in tokens:
        if w not in word2index:
            continue
        row[word2index[w]] += 1
    V.append(row)

V = np.array(V)

print V.shape

print V
