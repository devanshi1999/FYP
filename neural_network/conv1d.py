import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

def max_pooling(input):
  return T.max(input, axis=2)

def k_max_pooling(input, kmax):
  nbatches, nchannels, nwords, ndim = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
  x = input.dimshuffle(0,1,3,2)
  neighborsArgSorted = T.argsort(x, axis=3)
  ax0 = T.repeat(T.arange(nbatches), nchannels*ndim*kmax)
  ax1 = T.repeat(T.arange(nchannels), ndim * kmax).dimshuffle('x', 0)
  ax1 = T.repeat(ax1, nbatches, axis=0).flatten()
  ax2 = T.repeat(T.arange(ndim), kmax, axis=0).dimshuffle('x', 'x', 0)
  ax2 = T.repeat(ax2, nchannels, axis=1)
  ax2 = T.repeat(ax2, nbatches, axis=0).flatten()
  ax3 = T.sort(neighborsArgSorted[:,:,:,-kmax:], axis=3).flatten()

  pooled_out = x[ax0, ax1, ax2, ax3]
  pooled_out = pooled_out.reshape((nbatches, nchannels, ndim, kmax)).dimshuffle(0,1,3,2)
  return pooled_out
