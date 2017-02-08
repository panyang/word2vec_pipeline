import os
from gensim.models.doc2vec import Doc2Vec
from utils.mapreduce import corpus_iterator
from utils.data_utils import load_ORG_data
from tqdm import tqdm

import gensim.models
import psutil
from random import shuffle

import gensim.models.doc2vec


CPU_CORES = 1 #psutil.cpu_count()
assert (gensim.models.doc2vec.FAST_VERSION > -1)

class d2v_embedding(corpus_iterator):

    def __init__(self, *args, **kwargs):
        super(d2v_embedding, self).__init__(*args, **kwargs)
        self.epoch_n = int(kwargs["epoch_n"])

        self.clf = Doc2Vec(
            workers=CPU_CORES,
            dm=0,
            dbow_words=1,
            window=int(kwargs["window"]),
            negative=int(kwargs["negative"]),
            sample=float(kwargs["sample"]),
            size=int(kwargs["size"]),
            min_count=int(kwargs["min_count"])
        )

        self.data = None
        self.shuffle = kwargs["shuffle_during_epochs"]

    def preload_dataset(self, **config):
        print("Preloading datset into memory.")
        use_per_document_labels = config["use_per_document_labels"]
        cols = config["tagged_columns"]

        if not cols and not use_per_document_labels:
            msg = "Either use per_document_labels or choose columns for doc2vec"
            raise SyntaxError(msg)

        if cols:
            df = load_ORG_data(cols)
        
        TaggedDocument = gensim.models.doc2vec.TaggedDocument
        target_column = config["target_column"]

        self.data = []
        
        for row in self: 
            text = unicode(row[target_column])

            labels = []
            
            if cols:
                item = df.ix[int(row["_ref"])]
                for key in cols:
                    name = unicode(item[key]).replace(' ','_')
                    name = u'_{}_{}'.format(key,name)
                    labels.append(name)
                    
            if use_per_document_labels:
                labels.append(int(row["_ref"]))
                
            for sentence in text.split('\n'):
                tokens = sentence.split()
                val = TaggedDocument(tokens, labels)
                self.data.append(val)

    def get_data(self):
        if self.data is None:
            raise NotImplementedError("For now, data must be preloaded")

        if self.shuffle:
            shuffle(self.data)
        
        return self.data

    def compute(self, **config):
        print("Learning the vocabulary")

        self.clf.build_vocab(self.get_data())

        print("Training the features")
        for n in tqdm(range(self.epoch_n)):
            self.clf.train(self.get_data())

        print("Reducing the features")
        self.clf.init_sims(replace=True)

        print("Saving the features")
        out_dir = config["output_data_directory"]
        f_features = os.path.join(out_dir, config["d2v_embedding"]["f_db"])
        self.clf.save(f_features)
