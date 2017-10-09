import argparse

import skorch
import torch
from sklearn.model_selection import GridSearchCV

import data
from model import RNNModel
from learner import Learner

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# TODO: set seed

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

class LRAnnealing(skorch.callbacks.Callback):
    def on_epoch_end(self, net, **kwargs):
        if not net.history[-1]['valid_loss_best']:
            net.lr /= 4.0

class Checkpointing(skorch.callbacks.Callback):
    def on_epoch_end(self, net, **kwargs):
        if net.history[-1]['valid_loss_best']:
            net.save_params(args.save)

class ExamplePrinter(skorch.callbacks.Callback):
    def on_epoch_end(self, net, **kwargs):
        seed_sentence = "the meaning of"
        indices = [corpus.dictionary.word2idx[n] for n in seed_sentence.split()]
        indices = skorch.utils.to_var(torch.LongTensor([indices]).t(), use_cuda=args.cuda)
        sentence, _ = net.sample_n(num_words=10, input=indices)
        print(seed_sentence,
              " ".join([corpus.dictionary.idx2word[n] for n in sentence]))

def train_split(X, y):
    return X, corpus.valid, None, None

learner = Learner(
    module=RNNModel,
    max_epochs=args.epochs,
    batch_size=args.batch_size,
    use_cuda=args.cuda,
    callbacks=[LRAnnealing(), Checkpointing(), ExamplePrinter()],
    module__rnn_type='LSTM',
    module__ntoken=ntokens,
    module__ninp=200,
    module__nhid=200,
    module__nlayers=2,
    train_split=train_split,
    iterator_train=data.Loader,
    iterator_train__use_cuda=args.cuda,
    iterator_train__bptt=args.bptt,
    iterator_test=data.Loader,
    iterator_test__use_cuda=args.cuda,
    iterator_test__bptt=args.bptt)

# NOFIXME: iterator_test does not use corpus.valid as dataset
# REASON: we use GridSearchCV to generate validation splits
# FIXME: but we need validation data during training (LR annealing)

# FIXME: currently we have iterators for training and validation. Both of those
# supply (X,y) pairs. We do, however, also use the validation generator in
# predict (thus in scoring as well). Therefore we always generate `y` values
# even though we don't need to.

# TODO: easy way to write own score() that accesses the validation data only.

params = [
    {
        'lr': [10,20,30],
    },
]

pl = GridSearchCV(learner, params)
pl.fit(corpus.train)

print("Results of grid search:")
print("Best parameter configuration:", pl.best_params_)
print("Achieved score:", pl.best_score_)
