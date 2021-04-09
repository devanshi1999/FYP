import re
import os
import numpy as np
import cPickle
import subprocess
from collections import defaultdict

from stopwords import wordlist
from vocabulary import Vocabulary

UNKNOWN_WORD_IDX=0

def load_data(fname):
    #parsing data from xml files to retreive qid, question, correct and incorrect answers
    lines = open(fname).readlines()
    qids, questions, answers, labels = [], [], [], []
    num_skipped = 0
    previous_line = ''

    qid2num_answers = {}  #to store number of answers for a question or qid

    for i, line in enumerate(lines):
        line = line.strip()

        qid_match = re.match('<QApairs id=[\'|\"](.*)[\'|\"]>', line)

        # getting qid
        if qid_match:
            qid = qid_match.group(1)
            qid2num_answers[qid] = 0

        #getting question
        if previous_line and previous_line.startswith('<question>'):
            question = line.lower().split(' ')

        #getting answers and labels viz positive - 1, negative - 0
        label = re.match('<(positive|negative)>', previous_line)
        if label:
            label = label.group(1)
            label = 1 if label == 'positive' else 0
            answer = line.lower().split(' ')
            labels.append(label)
            answers.append(answer)
            questions.append(question)
            qids.append(qid)
            qid2num_answers[qid] += 1
        previous_line = line

    # print sorted(qid2num_answers.items(), key=lambda x: float(x[0]))
    return qids, questions, answers, labels

def compute_dfs(docs):
    word2df = defaultdict(float)
    for doc in docs:
        for w in set(doc):
            word2df[w] += 1.0
    num_docs = len(docs)
    
    for w, value in word2df.iteritems():
        word2df[w] /= (np.math.log(num_docs / value)+ 1)
    return word2df

def add_to_vocab(data, vocabulary):
    for sentence in data:
        for word in sentence:
            vocabulary.add(word)

def compute_overlap_features(questions, answers, word2df, stoplist=None):
    stoplist = stoplist if stoplist else set()
    feats_overlap = []
    for question, answer in zip(questions, answers):
        q_set = set([question_word for question_word in question if question_word not in stoplist])
        a_set = set([answer_word for answer_word in answer if answer_word not in stoplist])
        #overlapping words
        word_overlap = q_set.intersection(a_set)
        overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))

    #document frequency overlap
    df_overlap = 0.0
    for w in word_overlap:
      df_overlap += word2df[w]
    df_overlap /= (len(q_set) + len(a_set))

    feats_overlap.append(np.array([
                         overlap,
                         df_overlap,
                         ]))
    return np.array(feats_overlap)

def compute_overlap_idx(questions, answers, stoplist, q_max_length, a_max_length):
    stoplist = stoplist if stoplist else []
    feats_overlap = []
    q_indices, a_indices = [], []
    for question, answer in zip(questions, answers):
        q_set = set([question_word for question_word in question if question_word not in stoplist])
        a_set = set([answer_word for answer_word in answer if answer_word not in stoplist])
        word_overlap = q_set.intersection(a_set)

        q_idx = np.ones(q_max_length) * 2
        for i, q in enumerate(question):
            value = 0
            if q in word_overlap:
                value = 1
            q_idx[i] = value
        q_indices.append(q_idx)

        a_idx = np.ones(a_max_length) * 2
        for i, a in enumerate(answer):
            value = 0
            if a in word_overlap:
                value = 1
            a_idx[i] = value
        a_indices.append(a_idx)

    q_indices = np.vstack(q_indices).astype('int32')
    a_indices = np.vstack(a_indices).astype('int32')

    return q_indices, a_indices

def convert2indices(data, vocabulary, dummy_word_idx, max_length=40):
    data_idx = []
    for sentence in data:
        ex = np.ones(max_length) * dummy_word_idx
        for i, word in enumerate(sentence):
            idx = vocabulary.get(word, UNKNOWN_WORD_IDX)
            ex[i] = idx
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('int32')
    return data_idx
    
if __name__ == "__main__":
    stoplist = wordlist
    test = "/Users/devanshisingh/Desktop/FYP/text_files/Jacana/test.xml"
    fname = "/Users/devanshisingh/Desktop/FYP/text_files/test.txt"
    files = ' '.join([test])
    subprocess.call("/bin/cat {} > {}".format(files, fname), shell=True)

    qids, questions, answers, labels = load_data(fname)

    seen_qid = set()
    unique_questions = []
    for q, qid in zip(questions, qids):
        if qid not in seen_qid:
            seen_qid.add(qid)
            unique_questions.append(q)
    
    docs = answers + unique_questions

    word2dfs = compute_dfs(docs)

    vocabulary = Vocabulary(start_feature_id=0)
    vocabulary.add('UNKNOWN_WORD_IDX')
    add_to_vocab(answers, vocabulary)
    add_to_vocab(questions, vocabulary)

    cPickle.dump(vocabulary, open('/Users/devanshisingh/Desktop/FYP/parsed_files/vocab.pickle', 'w'))

    dummy_word_idx = vocabulary.fid

    q_max_length = 33
    a_max_length = 40

    overlap_feats = compute_overlap_features(questions, answers, stoplist=None, word2df=word2dfs)
    overlap_feats_stoplist = compute_overlap_features(questions, answers, stoplist=stoplist, word2df=word2dfs)
    overlap_feats = np.hstack([overlap_feats, overlap_feats_stoplist])

    qids = np.array(qids)
            
    labels = np.array(labels).astype('int32')
    
    _, counts = np.unique(labels, return_counts=True)

    q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_length, a_max_length)

    questions_idx = convert2indices(questions, vocabulary, dummy_word_idx, q_max_length)
    answers_idx = convert2indices(answers, vocabulary, dummy_word_idx, a_max_length)

    np.save('/Users/devanshisingh/Desktop/FYP/parsed_files/test.qids.npy', qids)
    np.save('/Users/devanshisingh/Desktop/FYP/parsed_files/test.questions.npy', questions_idx)
    np.save('/Users/devanshisingh/Desktop/FYP/parsed_files/test.answers.npy', answers_idx)
    np.save('/Users/devanshisingh/Desktop/FYP/parsed_files/test.q_overlap_indices.npy', q_overlap_indices)
    np.save('/Users/devanshisingh/Desktop/FYP/parsed_files/test.a_overlap_indices.npy', a_overlap_indices)
    np.save('/Users/devanshisingh/Desktop/FYP/parsed_files/test.labels.npy', labels)
