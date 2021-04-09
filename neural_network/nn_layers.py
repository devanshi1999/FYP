import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import theano.sandbox.neighbours as TSN
from theano.tensor.shared_randomstreams import RandomStreams

import conv1d

def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    """ 
    Shared variables are used to store neural network weights as we want these values to remain around across many executions of training or testing function. 
    Often, the purpose of a Theano training function is to update the weights stored in a shared variable. 
    And a testing function needs the current weights to perform the network's forward pass.
    
    """
    return theano.shared(value=numpy.zeros(shape, dtype=theano.config.floatX),
            name=name, borrow=True)

class Layer(object):
    def __init__(self):
        self.params = []
        self.weights = []
        self.biases = []

    def output_func(self, input):
        raise NotImplementedError("Each concrete class needs to implement output_func")

    def set_input(self, input):
        self.output = self.output_func(input)

    def __repr__(self):
        return "{}".format(self.__class__.__name__)

class FeedForwardNet(Layer):
    def __init__(self, layers=None, name=None):
        super(FeedForwardNet, self).__init__()

        if not name:
            name = self.__class__.__name__
        self.name = name

        self.layers = layers if layers else []
        for layer in layers:
            self.weights.extend(layer.weights)
            self.biases.extend(layer.biases)
            self.params.extend(layer.weights + layer.biases)
        self.num_params = sum([numpy.prod(p.shape.eval()) for p in self.params])

    def output_func(self, input):
        cur_input = input
        for layer in self.layers:
            layer.set_input(cur_input)
            cur_input = layer.output
        return cur_input

    def __repr__(self):
        layers_str = '\n'.join(['\t{}'.format(line) for layer in self.layers for line in str(layer).splitlines()])
        return '{} [num params: {}]\n{}'.format(self.name, self.num_params, layers_str)


class LookupTableFast(Layer):
    """ Basic linear transformation layer (W.X + b).
    Padding is used to force conv2d with valid mode behave as working in full mode."""
    def __init__(self, W=None, pad=None):
      super(LookupTableFast, self).__init__()
      self.pad = pad
      self.W = theano.shared(value=W, name='W_emb', borrow=True)
      self.weights = [self.W]

    def output_func(self, input):
      out = self.W[input.flatten()].reshape((input.shape[0], 1, input.shape[1], self.W.shape[1]))
      if self.pad:
        pad_matrix = T.zeros((out.shape[0], out.shape[1], self.pad, out.shape[3]))
        out = T.concatenate([pad_matrix, out, pad_matrix], axis=2)
      return out

    def __repr__(self):
      return "{}: {}".format(self.__class__.__name__, self.W.shape.eval())

class LookupTableFastStatic(Layer):
    """ Basic linear transformation layer (W.X + b).
    Padding is used to force conv2d with valid mode behave as working in full mode."""
    def __init__(self, W=None, pad=None):
      super(LookupTableFastStatic, self).__init__()
      self.pad = pad
      self.W = theano.shared(value=W, name='W_emb', borrow=True)

    def output_func(self, input):
      out = self.W[input.flatten()].reshape((input.shape[0], 1, input.shape[1], self.W.shape[1]))
      if self.pad:
        pad_matrix = T.zeros((out.shape[0], out.shape[1], self.pad, out.shape[3]))
        out = T.concatenate([pad_matrix, out, pad_matrix], axis=2)
      return out

    def __repr__(self):
      return "{}: {}".format(self.__class__.__name__, self.W.shape.eval())

class ParallelLookupTable(FeedForwardNet):
    def output_func(self, x):
        layers_out = []
        assert len(x) == len(self.layers)
        for x, layer in zip(x, self.layers):
            layer.set_input(x)
            layers_out.append(layer.output)
        return T.concatenate(layers_out, axis=3)


class ConvolutionLayer(Layer):
  """ Basic linear transformation layer (W.X + b) """
  def __init__(self, rng, filter_shape, input_shape=None, W=None):
        super(ConvolutionLayer, self).__init__()
        # initialize weights with random weights
        if W is None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
            W_bound = numpy.sqrt(1. / fan_in)
            W_data = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)

            # Simple initialization
            W = theano.shared(W_data, name="W_conv1d", borrow=True)

        self.filter_shape = filter_shape
        self.input_shape = input_shape

        self.W = W
        self.weights = [self.W]
  def __repr__(self):
        return "{}: filter_shape={}; input_shape={}".format(self.__class__.__name__, self.W.shape.eval(), self.input_shape)

class Conv2dLayer(ConvolutionLayer):

    def output_func(self, input):
        return conv.conv2d(input, self.W, border_mode='valid', filter_shape=self.filter_shape, image_shape=self.input_shape)

class NonLinearityLayer(Layer):
  def __init__(self, b_size, b=None, activation=T.tanh):
    super(NonLinearityLayer, self).__init__()
    if not b:
      b_values = numpy.zeros(b_size, dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)
    self.b = b
    self.activation = activation
    # In input we get a tensor (batch_size, nwords, ndim)
    self.biases = [self.b]

  def output_func(self, input):
    return self.activation(input + self.b.dimshuffle('x', 0, 'x', 'x'))

  def __repr__(self):
    return "{}: b_shape={} activation={}".format(self.__class__.__name__, self.b.shape.eval(), self.activation)

class KMaxPoolLayer(Layer):
  """Folds across last axis (ndim)."""
  def __init__(self, k_max):
    super(KMaxPoolLayer, self).__init__()
    self.k_max = k_max

  def output_func(self, input):
    # In input we get a tensor (batch_size, nwords, ndim)
    if self.k_max == 1:
      return conv1d.max_pooling(input)
    return conv1d.k_max_pooling(input, self.k_max)

  def __repr__(self):
    return "{}: k_max={}".format(self.__class__.__name__, self.k_max)


class FlattenLayer(Layer):
  """ Basic linear transformation layer (W.X + b) """

  def output_func(self, input):
    return input.flatten(2)

class ParallelLayer(FeedForwardNet):

  def output_func(self, input):
    layers_out = []
    for layer in self.layers:
      layer.set_input(input)
      layers_out.append(layer.output)
    return T.concatenate(layers_out, axis=1)

class PairwiseNoFeatsLayer(Layer):
  def __init__(self, q_in, a_in, activation=T.tanh):
    super(PairwiseNoFeatsLayer, self).__init__()

    W = build_shared_zeros((q_in, a_in), 'W_softmax_pairwise')

    self.W = W
    self.weights = [self.W]

  def __repr__(self):
    return "{}: W={}".format(self.__class__.__name__, self.W.shape.eval())

  def output_func(self, input):
      # P(Y|X) = softmax(W.X + b)
      q, a = input[0], input[1]
      # dot = T.batched_dot(q, T.batched_dot(a, self.W))
      dot = T.batched_dot(q, T.dot(a, self.W.T))
      out = T.concatenate([dot.dimshuffle(0, 'x'), q, a], axis=1)
      return out

class LinearLayer(Layer):
  def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh):
    super(LinearLayer, self).__init__()

    if W is None:
      W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)

      W = theano.shared(value=W_values, name='W', borrow=True)
    if b is None:
      b = build_shared_zeros((n_out,), 'b')

    self.W = W
    self.b = b

    self.activation = activation

    self.weights = [self.W]
    self.biases = [self.b]

  def output_func(self, input):
    return self.activation(T.dot(input, self.W) + self.b)

  def __repr__(self):
    return "{}: W_shape={} b_shape={} activation={}".format(self.__class__.__name__, self.W.shape.eval(), self.b.shape.eval(), self.activation)

class LogisticRegression(Layer):
    """Multi-class Logistic Regression
    """
    def __init__(self, n_in, n_out, W=None, b=None):
      if not W:
        W = build_shared_zeros((n_in, n_out), 'W_softmax')
      if not b:
        b = build_shared_zeros((n_out,), 'b_softmax')

      self.W = W
      self.b = b
      self.weights = [self.W]
      self.biases = [self.b]

    def __repr__(self):
      return "{}: W={}, b={}".format(self.__class__.__name__, self.W.shape.eval(), self.b.shape.eval())

    def output_func(self, input):
        # Calculate probability using softmax
        p_y_given_x = T.nnet.softmax(self._dot(input, self.W) + self.b)
        self.p_y_given_x = p_y_given_x
        # return the most probable answer
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        return self.y_pred

    def _dot(self, a, b):
        return T.dot(a, b)

    def negative_log_likelihood(self, y):
        '''
        Smaller the value better the classifier's confidence, so aim is to minimize this function
        '''
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood(y)

    def training_cost_weighted(self, y, weights=None):
        """ Wrapper for standard name """
        LL = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        weights = T.repeat(weights.dimshuffle('x', 0), y.shape[0], axis=0)
        factors = weights[T.arange(y.shape[0]), y]
        return -T.mean(LL * factors)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))

    def f1_score(self, y, labels=[0, 2]):
      """
      Mean F1 score between two classes (positive and negative as specified by the labels array).
      """
      y_tr = y
      y_pr = self.y_pred

      correct = T.eq(y_tr, y_pr)
      wrong = T.neq(y_tr, y_pr)

      label = labels[0]
      tp_neg = T.sum(correct * T.eq(y_tr, label))
      fp_neg = T.sum(wrong * T.eq(y_pr, label))
      fn_neg = T.sum(T.eq(y_tr, label) * T.neq(y_pr, label))
      tp_neg = T.cast(tp_neg, theano.config.floatX)
      prec_neg = tp_neg / T.maximum(1, tp_neg + fp_neg)
      recall_neg = tp_neg / T.maximum(1, tp_neg + fn_neg)
      f1_neg = 2. * prec_neg * recall_neg / T.maximum(1, prec_neg + recall_neg)

      label = labels[1]
      tp_pos = T.sum(correct * T.eq(y_tr, label))
      fp_pos = T.sum(wrong * T.eq(y_pr, label))
      fn_pos = T.sum(T.eq(y_tr, label) * T.neq(y_pr, label))
      tp_pos = T.cast(tp_pos, theano.config.floatX)
      prec_pos = tp_pos / T.maximum(1, tp_pos + fp_pos)
      recall_pos = tp_pos / T.maximum(1, tp_pos + fn_pos)
      f1_pos = 2. * prec_pos * recall_pos / T.maximum(1, prec_pos + recall_pos)

      return 0.5 * (f1_pos + f1_neg) * 100

    def accuracy(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        return T.mean(T.eq(self.y_pred, y)) * 100

class FastDropoutLayer(Layer):
    """ Basic linear transformation layer (W.X + b) """
    def __init__(self, rng):
      super(FastDropoutLayer, self).__init__()
      seed = rng.randint(2 ** 30)
      self.srng = RandomStreams(seed)

    def output_func(self, input):
      mask = self.srng.normal(size=input.shape, avg=1., dtype=theano.config.floatX)
      return input * mask

    def __repr__(self):
      return "{}".format(self.__class__.__name__)
