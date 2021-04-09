import sys
import numpy
import os
import theano
from theano import tensor as T
from datetime import datetime
import cPickle
from tqdm import tqdm
import time
from sklearn import metrics
from collections import defaultdict
import subprocess
import pandas as pd

from network_constants import *
import nn_layers
import sgd_trainer

import warnings
warnings.filterwarnings("ignore")

ZEROUT_DUMMY_WORD = True


mode = 'train'
if len(sys.argv) > 1:
    mode = sys.argv[1]
    print mode
    if not mode in ['TRAIN', 'TRAIN-ALL']:
        print "ERROR! The two possible training settings are: ['TRAIN', 'TRAIN-ALL']"
        sys.exit(1)

print "Running training in the {} setting".format(mode)

dataset_dir=mode

##### saving dataset files as per the training mode #########################################
# TRAINING AND OVERLAP FEATURES
if mode in ['TRAIN-ALL']:
    q_train = numpy.load(os.path.join(dataset_dir, 'train-all.questions.npy'))
    a_train = numpy.load(os.path.join(dataset_dir, 'train-all.answers.npy'))
    q_overlap_train = numpy.load(os.path.join(dataset_dir, 'train-all.q_overlap_indices.npy'))
    a_overlap_train = numpy.load(os.path.join(dataset_dir, 'train-all.a_overlap_indices.npy'))
    y_train = numpy.load(os.path.join(dataset_dir, 'train-all.labels.npy'))
else:
    q_train = numpy.load(os.path.join(dataset_dir, 'train.questions.npy'))
    a_train = numpy.load(os.path.join(dataset_dir, 'train.answers.npy'))
    q_overlap_train = numpy.load(os.path.join(dataset_dir, 'train.q_overlap_indices.npy'))
    a_overlap_train = numpy.load(os.path.join(dataset_dir, 'train.a_overlap_indices.npy'))
    y_train = numpy.load(os.path.join(dataset_dir, 'train.labels.npy'))

# DEV/TEST
q_dev = numpy.load(os.path.join(dataset_dir, 'dev.questions.npy'))
a_dev = numpy.load(os.path.join(dataset_dir, 'dev.answers.npy'))
q_overlap_dev = numpy.load(os.path.join(dataset_dir, 'dev.q_overlap_indices.npy'))
a_overlap_dev = numpy.load(os.path.join(dataset_dir, 'dev.a_overlap_indices.npy'))
y_dev = numpy.load(os.path.join(dataset_dir, 'dev.labels.npy'))
qids_dev = numpy.load(os.path.join(dataset_dir, 'dev.qids.npy'))

q_test = numpy.load(os.path.join(dataset_dir, 'test.questions.npy'))
a_test = numpy.load(os.path.join(dataset_dir, 'test.answers.npy'))
q_overlap_test = numpy.load(os.path.join(dataset_dir, 'test.q_overlap_indices.npy'))
a_overlap_test = numpy.load(os.path.join(dataset_dir, 'test.a_overlap_indices.npy'))
y_test = numpy.load(os.path.join(dataset_dir, 'test.labels.npy'))
qids_test = numpy.load(os.path.join(dataset_dir, 'test.qids.npy'))


print 'y_train', numpy.unique(y_train, return_counts=True)
print 'y_dev', numpy.unique(y_dev, return_counts=True)
print 'y_test', numpy.unique(y_test, return_counts=True)

print 'q_train', q_train.shape
print 'q_dev', q_dev.shape
print 'q_test', q_test.shape

print 'a_train', a_train.shape
print 'a_dev', a_dev.shape
print 'a_test', a_test.shape

numpy_rng = numpy.random.RandomState(123)
q_max_size = q_train.shape[1]          # max words in a question sentence
a_max_size = a_train.shape[1]          # max words in a answer sentence


dim = 5
print "Generating random vocabulary for word overlap indicator features with dim:", dim
dummy_word_id = numpy.max(a_overlap_train)
#Gaussian distribution
vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, dim) * 0.25
vocab_emb_overlap[-1] = 0

#loding Glove word embeddings
fname = os.path.join(dataset_dir, 'emb_glove.6B.50d.txt.npy')
vocab_emb = numpy.load(fname)        
ndim = vocab_emb.shape[1]
dummy_word_id = numpy.max(a_train)
print dummy_word_id

#input sentence matrices
x = T.dmatrix('x')
x_q = T.lmatrix('q')
x_q_overlap = T.lmatrix('q_overlap')
x_a = T.lmatrix('a')
x_a_overlap = T.lmatrix('a_overlap')
y = T.ivector('y')

########################## CONVOLUTION LAYER ########################

ndim = vocab_emb.shape[1] + vocab_emb_overlap.shape[1]    # 50 + 5 = 55
activation = T.tanh

### CNN FOR QUESTION ###

lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths)-1)
lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths)-1)
lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])

num_input_channels = 1
input_shape = (batch_size, num_input_channels, q_max_size + 2*(max(q_filter_widths)-1), ndim)

conv_layers = []

# each conv_layer consists of 2d convolution , filters, activation, pooling layers
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



### CNN FOR ANSWERS ###

lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths)-1)
lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths)-1)
lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap])

num_input_channels = 1
input_shape = (batch_size, num_input_channels, a_max_size + 2*(max(a_filter_widths)-1), ndim)

conv_layers = []

# each conv_layer consists of 2d convolution , filters, activation, pooling layers
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

dropout_q = nn_layers.FastDropoutLayer(rng=numpy_rng)
dropout_a = nn_layers.FastDropoutLayer(rng=numpy_rng)
dropout_q.set_input(nnet_q.output)
dropout_a.set_input(nnet_a.output)

# QA Pair Matching Layer
# using equation sim = (nnet_q.output).M.(nnet_a.output)
pairwise_layer = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in,a_in=a_logistic_n_in)
# pairwise_layer.set_input((dropout_q.output, dropout_a.output))
pairwise_layer.set_input((nnet_q.output, nnet_a.output))


#################### HIDDEN LAYER AND FINAL CLASSIFIER ###################
# no of inputs for hidden layer
n_in = q_logistic_n_in + a_logistic_n_in + 1

hidden_layer = nn_layers.LinearLayer(numpy_rng, n_in=n_in, n_out=n_in, activation=activation)
hidden_layer.set_input(pairwise_layer.output)

classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
classifier.set_input(hidden_layer.output)

# Final Neural Network to be trained
train_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, pairwise_layer, hidden_layer, classifier],name="Training nnet")
test_nnet = train_nnet


#################### TRAINING THE NETWORK #############################

ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')

# output dump directory
# nnet_outdir = 'exp.out/ndim={};batch={};max_norm={};learning_rate={};{}'.format(ndim, batch_size, max_norm, learning_rate, ts)
# if not os.path.exists(nnet_outdir):
#     os.makedirs(nnet_outdir)
# nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
# print "Saving to", nnet_fname
# cPickle.dump([train_nnet, test_nnet], open(nnet_fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

# network parameters - Weights and Biases
params = train_nnet.params

'''
total_params = sum([numpy.prod(param.shape.eval()) for param in params])
print 'Total params number:', total_params

# training cost value - Negative Log Likelihood (NLL) 
cost = train_nnet.layers[-1].training_cost(y)

# Predictions made and their associated probabilities
predictions = test_nnet.layers[-1].y_pred
predictions_prob = test_nnet.layers[-1].p_y_given_x[:,-1]

# batch input theano variables
batch_x_q = T.lmatrix('batch_x_q')
batch_x_a = T.lmatrix('batch_x_a')
batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')
batch_x_a_overlap = T.lmatrix('batch_x_a_overlap')
batch_y = T.ivector('batch_y')

# optimizer - Stochastic Gradient Descent - Adadelta Algo
updates = sgd_trainer.get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=max_norm, word_vec_name='W_emb')

# defining input variables
inputs_test = [batch_x_q,
                batch_x_a,
                batch_x_q_overlap,
                batch_x_a_overlap,
                ]
givens_test = {x_q: batch_x_q,
                x_a: batch_x_a,
                x_q_overlap: batch_x_q_overlap,
                x_a_overlap: batch_x_a_overlap,
                }
inputs_train = [batch_x_q,
                batch_x_a,
                batch_x_q_overlap,
                batch_x_a_overlap,
                batch_y,
                ]
givens_train = {x_q: batch_x_q,
                x_a: batch_x_a,
                x_q_overlap: batch_x_q_overlap,
                x_a_overlap: batch_x_a_overlap,
                y: batch_y}


labels = sorted(numpy.unique(y_test))
print 'Output labels for Binary Classification', labels

print "Zero out dummy word:", ZEROUT_DUMMY_WORD
if ZEROUT_DUMMY_WORD:
    W_emb_list = [w for w in params if w.name == 'W_emb']
    zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

# theano function computes the value of cost using the given inputs for learning params while training
train_fn = theano.function(inputs=inputs_train,
                            outputs=cost,
                            updates=updates,
                            givens=givens_train)

# theano function computes the prediction value given the inputs while testing
pred_fn = theano.function(inputs=inputs_test,
                            outputs=predictions,
                            givens=givens_test)
pred_prob_fn = theano.function(inputs=inputs_test,
                            outputs=predictions_prob,
                        givens=givens_test)

def predict_prob_batch(batch_iterator):
    # return predicted probability P(y|x) for all samples
    preds = numpy.hstack([pred_prob_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
    return preds[:batch_iterator.n_samples]

def map_score(qids, labels, probs):
    # calculate MAP - Mean Average Precision Score
    qid2cand = defaultdict(list)
    for qid, label, prob in zip(qids, labels, probs):
        qid2cand[qid].append((prob, label))

    avg_precisions = []
    for qid, candidates in qid2cand.iteritems():
        avg_precision = 0
        running_correct_count = 0
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            if label > 0:
                running_correct_count += 1
                avg_precision += float(running_correct_count) / i
        avg_precisions.append(avg_precision / (running_correct_count + 1e-6))
    map_score = sum(avg_precisions) / len(avg_precisions)
    return map_score

# iterators for each data-set batches 
train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_train, a_train, q_overlap_train, a_overlap_train, y_train], batch_size=batch_size, randomize=True)
dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_dev, a_dev, q_overlap_dev, a_overlap_dev, y_dev], batch_size=batch_size, randomize=False)
test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, y_test], batch_size=batch_size, randomize=False)

##############

epoch = 0
best_dev_acc = -numpy.inf
num_train_batches = len(train_set_iterator)
timer_train = time.time()
no_best_dev_update = 0

# Start Training
while epoch < n_epochs:
    # epoch start time
    timer = time.time()
    for i, (x_q, x_a, x_q_overlap, x_a_overlap, y) in enumerate(tqdm(train_set_iterator), 1):
        # calculate cost
        train_fn(x_q, x_a, x_q_overlap, x_a_overlap, y)

        if ZEROUT_DUMMY_WORD:
            zerout_dummy_word()
        
        # dev accuracy to filter out best model after every 10 batches
        if i % 10 == 0 or i == num_train_batches:
            y_pred_dev = predict_prob_batch(dev_set_iterator)
            dev_acc = metrics.roc_auc_score(y_dev, y_pred_dev) * 100
            if dev_acc > best_dev_acc:
                # test accuracy
                y_pred_probs = predict_prob_batch(test_set_iterator)
                test_acc = map_score(qids_test, y_test, y_pred_probs) * 100
                print('epoch: {} batch: {} dev acc: {:.4f}; test map: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc, test_acc, best_dev_acc))
                best_dev_acc = dev_acc
                best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                no_best_dev_update = 0

    if no_best_dev_update >= 3:
        print "Quitting after of no update of the best score on dev set", no_best_dev_update
        break

    epoch += 1
    no_best_dev_update += 1

############## CREATING AND SAVING DUMP FILES ######################################

print('Training took: {:.4f} seconds'.format(time.time() - timer_train))

# Learned Parameters
for i, param in enumerate(best_params):
    params[i].set_value(param, borrow=True)

y_pred_test = predict_prob_batch(test_set_iterator)
test_acc = map_score(qids_test, y_test, y_pred_test) * 100
fname = os.path.join(nnet_outdir, 'best_dev_params.epoch={:02d};batch={:05d};dev_acc={:.2f}.dat'.format(epoch, i, best_dev_acc))
numpy.savetxt(os.path.join(nnet_outdir, 'test.epoch={:02d};batch={:05d};dev_acc={:.2f}.predictions.npy'.format(epoch, i, best_dev_acc)), y_pred_test)
cPickle.dump(best_params, open(fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

print "Running trec_eval script..."
N = len(y_pred_test)

# preparing input files for TREC Eval Script
df_submission = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
df_submission['qid'] = qids_test
df_submission['iter'] = 0
df_submission['docno'] = numpy.arange(N)
df_submission['rank'] = 0
df_submission['sim'] = y_pred_test
df_submission['run_id'] = 'nnet'
df_submission.to_csv(os.path.join(nnet_outdir, 'submission.txt'), header=False, index=False, sep=' ')

df_gold = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
df_gold['qid'] = qids_test
df_gold['iter'] = 0
df_gold['docno'] = numpy.arange(N)
df_gold['rel'] = y_test
df_gold.to_csv(os.path.join(nnet_outdir, 'gold.txt'), header=False, index=False, sep=' ')

subprocess.call("/bin/sh run_eval_script.sh '{}'".format(nnet_outdir), shell=True)


'''