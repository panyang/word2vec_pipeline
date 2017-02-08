import os
from gensim.models.doc2vec import Doc2Vec
from utils.mapreduce import corpus_iterator
from tqdm import tqdm

import gensim.models
import psutil

CPU_CORES = 1 #psutil.cpu_count()
assert (gensim.models.doc2vec.FAST_VERSION > -1)

class d2v_embedding(corpus_iterator):

    def __init__(self, *args, **kwargs):
        super(d2v_embedding, self).__init__(*args, **kwargs)
        self.epoch_n = int(kwargs["epoch_n"])

        self.clf = Doc2Vec(
            workers=CPU_CORES,
            dm=0,
            window=int(kwargs["window"]),
            negative=int(kwargs["negative"]),
            sample=float(kwargs["sample"]),
            size=int(kwargs["size"]),
            min_count=int(kwargs["min_count"])
        )

        self.data = None


    def preload_dataset(self, **config):
        print("Preloading datset into memory.")
        LSI = self.labelized_sentence_iterator
        self.data = list(LSI(config["target_column"]))

    def compute(self, **config):
        print("Learning the vocabulary")

        if self.data is None:
            raise NotImplementedError("For now, data must be preloaded")
            #LSI = self.labelized_sentence_iterator       
            #ITR = LSI(config["target_column"])
            #self.clf.build_vocab(ITR)

        self.clf.build_vocab(self.data)

        print("Training the features")
        for n in tqdm(range(self.epoch_n)):
            self.clf.train(self.data)

        print("Reducing the features")
        self.clf.init_sims(replace=True)

        print("Saving the features")
        out_dir = config["output_data_directory"]
        f_features = os.path.join(out_dir, config["d2v_embedding"]["f_db"])
        self.clf.save(f_features)
