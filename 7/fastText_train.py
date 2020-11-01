from gensim.models import FastText
from gensim.models.word2vec import PathLineSentences
from gensim.models.callbacks import CallbackAny2Vec
import argparse
import time

parser = argparse.ArgumentParser(description='6 assignment')
parser.add_argument('--mode', type=int,default=1,help='1: train 2: eval')
parser.add_argument('--path', type=int,default=1,help='model name')
parser.add_argument('--size', type=int,default=300,help='dimension of vector')
parser.add_argument('--ws', type=int,default=5,help='window size; number context word/2')
parser.add_argument('--mc', type=int,default=5,help='min_count; min frequency')
parser.add_argument('--workers', type=int,default=4,help='number of threads')
parser.add_argument('--sg', type=int,default=1,help='1: skip gram')
parser.add_argument('--hs', type=int,default=0,help='1 for hierarchical soft,0 for negative sampling')
parser.add_argument('--negative', type=int,default=15,help='number of negative samples')
parser.add_argument('--cm', type=int,default=1,help='cbow_mean; 1 for avg, 0 for sum')
parser.add_argument('--ne', type=float,default=0.75,help='ns_exponent; unigram distribution')
parser.add_argument('--alpha', type=float,default=0.01,help='learning rate')
parser.add_argument('--ma', type=float,default=0.0001,help='min_alpha; learning rate decay -> min learning rate')
parser.add_argument('--iter', type=int,default=5,help='number of epoch')
parser.add_argument('--sb', type=int,default=1,help='1: sorted_vocab')
parser.add_argument('--bs', type=int,default=10000,help='batch_words; batch size')
parser.add_argument('--wn', type=int,default=1,help='subword')
parser.add_argument('--minn', type=int,default=3,help='n_gram min')
parser.add_argument('--maxn', type=int,default=6,help='n_gram max')

args = parser.parse_args()

class callback(CallbackAny2Vec): 
    """Callback to print each epoch.""" 
    def __init__(self): 
        self.epoch = 0 
        self.loss_to_be_subed = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch)) 
    
    def on_epoch_end(self, model): 
        print('epoch {} end'.format(self.epoch)) 
        self.epoch += 1

if args.mode==1:
    print('train')
    sentences = PathLineSentences("./data/1billion/training-monolingual.tokenized.shuffled/")
    model = FastText(sentences=sentences, size=args.size, window=args.ws, min_count=args.mc, workers=args.workers, 
                     sg=args.sg, hs=args.hs,negative=args.negative, ns_exponent=args.ne, 
                    alpha=args.alpha, min_alpha=args.ma, iter=args.iter,
                    word_ngrams=args.wn, min_n=args.minn, max_n=args.maxn,callbacks=[callback()])
    model.save("./saved_model/fastText"+str(args.path)+".model")
    print(len(model.wv.vocab))
    score, predictions = model.wv.evaluate_word_analogies('../6/data/questions-words.txt')
    print(score)

if args.mode==2:
    print('eval')
    model = FastText.load("./saved_model/fastText"+str(args.path)+".model")
    score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
    print(score)
    print(model.wv.most_similar("thank____you", topn=20))
    print(len(model.wv.vocab))


