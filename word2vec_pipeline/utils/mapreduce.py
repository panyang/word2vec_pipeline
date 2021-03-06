import gensim.models.doc2vec

LabeledSentence = gensim.models.doc2vec.LabeledSentence


class simple_mapreduce(object):

    def __init__(self, *args, **kwargs):
        # Set any function arguments for calling
        self.kwargs = kwargs

    def __call__(self, x):
        raise NotImplementedError

    def reduce(self, *x):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError


class corpus_iterator(simple_mapreduce):

    def set_iterator_function(self, iter_func, *args, **kwargs):
        self.iter_func = iter_func
        self.iter_args = args
        self.iter_kwargs = kwargs

    def __iter__(self):
        for x in self.iter_func(*self.iter_args, **self.iter_kwargs):
            yield x

    def sentence_iterator(self, target_column):
        for row in self:
            text = row[target_column]
            yield unicode(text).split()

    def labelized_sentence_iterator(self):
        for item in self:
            text = item[0]
            idx = item[1]
            f_sql = item[-1]

            for sentence in text.split('\n'):
                sentence = unicode(sentence)
                tokens = sentence.split()
                label = "{}_{}".format(f_sql, idx)
                yield LabeledSentence(tokens, [label, ])
