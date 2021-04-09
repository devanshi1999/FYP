import numpy as np
import cPickle
import os

def load_vectors(fname, words):

    ''' each vector is 1 x 50 dimensional pretrained Glove vector trained on

        Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased) '''

    vocab = set(words)
    word_vectors = {}
    vocab_size = 400000
    ndim=50

    file = open(fname,"r")
    Content = file.read() 
    lines = Content.split("\n")

    found=0
    random=0
    for line in lines:
        tokens = line.split(" ")
        word = tokens[0]
        if word in vocab:
            found+=1
            word_vectors[word] = np.array(tokens[1:],dtype='float32')
            # print word_vectors[word]
    
    print "extracting done, found", found, "words out of", len(vocab),"words in dataset"
    
    for word in vocab:
        word_vec = word_vectors.get(word, None)
        if word_vec is None:
            word_vectors[word] = np.random.uniform(-0.25, 0.25, ndim)
            # print word_vectors[word]
            random+=1
    
    print "assigned random vectors for", random, "words"
    return word_vectors, ndim


if __name__ == "__main__":
    np.random.seed(123)

    fname_vocab = '/Users/devanshisingh/Desktop/FYP/parsed_files/vocab.pickle'
    vocabulary = cPickle.load(open(fname_vocab))
    words = vocabulary.keys()

    fname ='/Users/devanshisingh/Desktop/FYP/embeddings/glove.6B.50d.txt'
    
    gloveVec, ndim = load_vectors(fname, words)

####### creating and saving in a file the vocabulary of these word embeddings
    embeddings_vocab = np.zeros((len(vocabulary) + 1, ndim))
    for word, idx in vocabulary.iteritems():
        embeddings_vocab[idx] = gloveVec[word]

    outfile = '/Users/devanshisingh/Desktop/FYP/parsed_files/emb.npy'
    print "embeddings are saved in", outfile
    np.save(outfile, embeddings_vocab)

    