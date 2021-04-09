from collections import OrderedDict
from nn_layers import build_shared_zeros
from theano import tensor as T
import numpy
import theano
import time
from fish import ProgressFish

def get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=9, word_vec_name='W_emb'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    print "Generating adadelta updates"
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        exp_sqr_grads[param] = build_shared_zeros(param.shape.eval(), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = build_shared_zeros(param.shape.eval(), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + eps) / T.sqrt(up_exp_sg + eps)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        # if (param.get_value(borrow=True).ndim == 2) and (param.name != word_vec_name):
        if max_norm and param.name != word_vec_name:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(max_norm))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

class MiniBatchIteratorConstantBatchSize(object):
    """ Basic mini-batch iterator """
    def __init__(self, rng, datasets, batch_size=100, randomize=False):
        self.rng = rng

        self.batch_size = batch_size
        self.n_samples = datasets[0].shape[0]           # no of QA pairs
        
        padded_datasets = []
        for d in datasets:
            # padding for case when #of data points % batch size <> 0
          pad_size = batch_size - len(d) % batch_size
          pad = d[:pad_size]
          padded_dataset = numpy.concatenate([d, pad])
          padded_datasets.append(padded_dataset)
        self.datasets = padded_datasets
        self.n_batches = (self.n_samples + self.batch_size - 1) / self.batch_size
        # self.n_batches = self.n_samples / self.batch_size

        self.randomize = randomize
        # print 'n_samples', self.n_samples
        # print 'n_batches', self.n_batches

    def __len__(self):
      return self.n_batches

    def __iter__(self):
        n_batches = self.n_batches
        batch_size = self.batch_size
        n_samples = self.n_samples
        if self.randomize:
            for _ in xrange(n_batches):
              i = self.rng.randint(n_batches)
              yield [x[i*batch_size:(i+1)*batch_size] for x in self.datasets]
        else:
            for i in xrange(n_batches):
              yield [x[i*batch_size:(i+1)*batch_size] for x in self.datasets]
