import numpy
import theano
from theano import tensor as T
import cPickle
import pandas as pd
import nn_layers 
from network_constants import *
import warnings
warnings.filterwarnings("ignore")

numpy_rng = numpy.random.RandomState(123)

######################### EXTRACTING EMBEDDED SENTENCE FILES ##########################
qids_test = numpy.load('/Users/devanshisingh/Desktop/FYP/parsed_files/test.qids.npy')
q_test = numpy.load('/Users/devanshisingh/Desktop/FYP/parsed_files/test.questions.npy')
a_test = numpy.load('/Users/devanshisingh/Desktop/FYP/parsed_files/test.answers.npy')
q_overlap_test = numpy.load('/Users/devanshisingh/Desktop/FYP/parsed_files/test.q_overlap_indices.npy')
a_overlap_test = numpy.load('/Users/devanshisingh/Desktop/FYP/parsed_files/test.a_overlap_indices.npy')
y_test = numpy.load('/Users/devanshisingh/Desktop/FYP/parsed_files/test.labels.npy')

x_q = T.lmatrix('q')
x_q_overlap = T.lmatrix('q_overlap')
x_a = T.lmatrix('a')
x_a_overlap = T.lmatrix('a_overlap')
y = T.ivector('y')

numpy_rng = numpy.random.RandomState(123)
q_max_size = 33         
a_max_size = 40 

batch_size = q_test.shape[0]
dim = 5
dummy_word_id = numpy.max(a_overlap_test)
vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, dim) * 0.25
vocab_emb_overlap[-1] = 0

fname = '/Users/devanshisingh/Desktop/FYP/parsed_files/emb.npy'
vocab_emb = numpy.load(fname)        
ndim = vocab_emb.shape[1]
dummy_word_id = numpy.max(a_test)

########################## CONVOLUTION LAYER ########################
ndim = vocab_emb.shape[1] + vocab_emb_overlap.shape[1]    # 50 + 5 = 55
activation = T.tanh

# Q CNN
lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths)-1)
lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths)-1)
lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])

num_input_channels = 1
input_shape = (batch_size, num_input_channels, q_max_size + 2*(max(q_filter_widths)-1), ndim)

conv_layers = []

for filter_width in q_filter_widths:
    filter_shape = (nfilters, num_input_channels, filter_width, ndim)
    conv = nn_layers.Conv2dLayer(rng=numpy_rng, filter_shape=filter_shape, input_shape=input_shape)
    non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=activation)
    pooling = nn_layers.KMaxPoolLayer(k_max=q_k_max)
    conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[conv, non_linearity, pooling])
    conv_layers.append(conv2dNonLinearMaxPool)

join_layer = nn_layers.ParallelLayer(layers=conv_layers)
flatten_layer = nn_layers.FlattenLayer()

nnet_q = nn_layers.FeedForwardNet(layers=[
                                lookup_table,
                                join_layer,
                                flatten_layer,
                                ])

nnet_q.set_input((x_q, x_q_overlap))

# A CNN
lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(a_filter_widths)-1)
lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(a_filter_widths)-1)
lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])

num_input_channels = 1
input_shape = (batch_size, num_input_channels, a_max_size + 2*(max(a_filter_widths)-1), ndim)

conv_layers = []

for filter_width in a_filter_widths:
    filter_shape = (nfilters, num_input_channels, filter_width, ndim)
    conv = nn_layers.Conv2dLayer(rng=numpy_rng, filter_shape=filter_shape, input_shape=input_shape)
    non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=activation)
    pooling = nn_layers.KMaxPoolLayer(k_max=a_k_max)
    conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[conv, non_linearity, pooling])
    conv_layers.append(conv2dNonLinearMaxPool)

join_layer = nn_layers.ParallelLayer(layers=conv_layers)
flatten_layer = nn_layers.FlattenLayer()

nnet_a = nn_layers.FeedForwardNet(layers=[
                                lookup_table,
                                join_layer,
                                flatten_layer,
                                ])

nnet_a.set_input((x_a, x_a_overlap))

######################### MATCHING LAYER ############################
q_logistic_n_in = nfilters * len(q_filter_widths) * q_k_max        # size of final intermediate representation
a_logistic_n_in = nfilters * len(a_filter_widths) * a_k_max

pairwise_layer = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in,a_in=a_logistic_n_in)
pairwise_layer.set_input((nnet_q.output, nnet_a.output))

#################### HIDDEN LAYER AND FINAL CLASSIFIER ###################
n_in = q_logistic_n_in + a_logistic_n_in + 1

hidden_layer = nn_layers.LinearLayer(numpy_rng, n_in=n_in, n_out=n_in, activation=activation)
hidden_layer.set_input(pairwise_layer.output)

classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
classifier.set_input(hidden_layer.output)

test_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, pairwise_layer, hidden_layer, classifier],name="Testing nnet")

##################### EXTRACTING THE LEARNT PARAMETERS #######################
# param_file = "exp.out/ndim=55;batch=10;max_norm=0;learning_rate=0.1;2021-03-09-19.38.17/best_dev_params.epoch=10;batch=00010;dev_acc=79.16.dat"
outdir = "/Users/devanshisingh/Desktop/FYP/neural_network/exp.out/"
param_file = outdir + "ndim=55;batch=10;max_norm=0;learning_rate=0.1;2021-03-09-19.38.17/best_dev_params.epoch=10;batch=00010;dev_acc=79.16.dat"

f = open(param_file,'rb')
best_params = cPickle.load(f)

for i, param in enumerate(best_params):
    test_nnet.params[i].set_value(param, borrow=True)

######################### TESTING ########################################
ZEROUT_DUMMY_WORD = True
if ZEROUT_DUMMY_WORD:
    W_emb_list = [w for w in test_nnet.params if w.name == 'W_emb']
    zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])
    zerout_dummy_word()

inputs_test = [x_q,
                x_a,
                x_q_overlap,
                x_a_overlap,
                ]
predictions_prob = test_nnet.layers[-1].p_y_given_x[:,-1]
pred_prob_fn = theano.function(inputs=inputs_test, outputs=predictions_prob)
preds = pred_prob_fn(q_test, a_test, q_overlap_test, a_overlap_test )

N = len(preds)


df_submission = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'sim'])
df_submission['qid'] = qids_test
df_submission['sim'] = preds
df_submission.to_csv('/Users/devanshisingh/Desktop/FYP/text_files/results.txt', header=False, index=False, sep=' ')
