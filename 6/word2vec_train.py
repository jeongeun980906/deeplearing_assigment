from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences
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
parser.add_argument('--sg', type=int,default=0,help='1: skip gram')
parser.add_argument('--hs', type=int,default=0,help='1 for hierarchical soft,0 for negative sampling')
parser.add_argument('--negative', type=int,default=15,help='number of negative samples')
parser.add_argument('--cm', type=int,default=1,help='cbow_mean; 1 for avg, 0 for sum')
parser.add_argument('--ne', type=float,default=0.75,help='ns_exponent; unigram distribution')
parser.add_argument('--alpha', type=float,default=0.01,help='learning rate')
parser.add_argument('--ma', type=float,default=0.0001,help='min_alpha; learning rate decay -> min learning rate')
parser.add_argument('--iter', type=int,default=5,help='number of epoch')
parser.add_argument('--sb', type=int,default=1,help='1: sorted_vocab')
parser.add_argument('--bs', type=int,default=10000,help='batch_words; batch size')
args = parser.parse_args()

class callback(CallbackAny2Vec): 
    """Callback to print loss after each epoch.""" 
    def __init__(self): 
        self.epoch = 0 
        self.loss_to_be_subed = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch)) 
    
    def on_epoch_end(self, model): 
        loss = model.get_latest_training_loss() 
        loss_now = loss - self.loss_to_be_subed 
        self.loss_to_be_subed = loss 
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now)) 
        self.epoch += 1
    
#-------------pre-trained word2vec---------------
# model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True, limit=60000)
# score, predictions = model.evaluate_word_analogies('data/questions-words.txt')

# print(score)
# #print(model['apple'])
# print("similarity between apple and fruit: {}".format(model.similarity("apple", "fruit")))
# print("similarity between apple and car: {}".format(model.similarity("apple", "car")))
# print(model.most_similar("apple", topn=10))
# print(model.most_similar(positive=['king', 'women'], negative=['man'], topn=10))

#-------------training---------------
if args.mode==1:
    sentences = PathLineSentences("./data/1billion/training-monolingual.tokenized.shuffled/")
    #sentences = LineSentence("./data/news1.txt")
    print('train starts')
    start_time = time.time()
    model = Word2Vec(sentences, size=args.size, window=args.ws, min_count=args.mc, workers=args.workers, sg=args.sg, hs=args.hs,
                 negative=args.negative, ns_exponent=args.ne, cbow_mean=args.cm, alpha=args.alpha, min_alpha=args.ma, iter=args.iter,
                 sorted_vocab=args.sb,batch_words=args.bs,compute_loss=True,callbacks=[callback()])
    end_time=time.time()
    training_time = (end_time - start_time) / 60
    print(len(model.wv.vocab))
    score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
    print('score:',score)
    ts=round(start_time,3)
    te=round(end_time,3)
    tr=round(training_time,3)
    print('train end',te,'train start',ts)
    print('training time',tr)
    model.save("./saved_model/word2vec"+str(args.path)+".model")

#-------------evaluation---------------
if args.mode==2:
    print('evaluation')
    model = Word2Vec.load("./saved_model/word2vec"+str(args.path)+".model")
    print('model loaded')
    score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
    print(score)

    print(model.wv.most_similar("car", topn=200))
    print(len(model.wv.vocab))
    print("similarity between apple and fruit: {}".format(model.wv.similarity("apple", "fruit")))
    print(model.wv["apple"])


