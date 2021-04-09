import jsonrpc
from json import loads
from pprint import pprint
import nltk
import codecs
import pandas as pd
import numpy

pronouns = ['PRP', 'PRP$']
nouns = ['NNP', 'NN', 'NNS', 'NNPS']

def coref_resolution(text):
    server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                             jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))

    result = loads(server.parse(text))
    return result
    # return result['coref']

def tokenize_text(text):
    token_sen = nltk.sent_tokenize(text)
    word = []
    for i in range(len(token_sen)):
        word.append(nltk.word_tokenize(token_sen[i]))
    return word

def coref_rephrase(text):
    coref = coref_resolution(text)
    if 'coref' not in coref.keys():
        return text
    coref = coref['coref']
    process_text = tokenize_text(text)
        
    for coref_entity in coref:
        for coref_entity_element in coref_entity:
       
            pos_tag_left = nltk.pos_tag([coref_entity_element[0][0]])
            pos_tag_right = nltk.pos_tag([coref_entity_element[1][0]])
            
            
            #print a[j[0][1]]
            #print pos_tag_left[0][0], " | ", pos_tag_left[0][1], " | ", pos_tag_right[0][0], " | ", pos_tag_right[0][1]
            #a being the word tokenizer
            if pos_tag_left[0][1] in pronouns and pos_tag_right[0][1] in nouns:
                if pos_tag_left[0][0] in process_text[coref_entity_element[0][1]]:
                
                   process_text[coref_entity_element[0][1]][process_text[coref_entity_element[0][1]].index(pos_tag_left[0][0])] = pos_tag_right[0][0]
                    
                
                           
    rephrase = [[' '.join(word) for word in process_text]]
    return rephrase

def divide_rephrase(text,fname):
    f=open(fname,'w')
    batch_size = 1023
    nbatches = int(numpy.ceil(float(len(text))/batch_size))
    print "batch_size:", batch_size, "no. of batches:", nbatches
    start = 0
    end = -1
    for i in range(nbatches):
        print "batch",i,"in corefAnnotator"
        start = end+1
        end = start+batch_size
        txt = ""
        if end >= len(text):
            txt = text[start:-1]
        else:
            txt = text[start:end]
        
        results = coref_rephrase(txt)
        for result in results:
            for sentence in result:
                s = str(sentence)
                f.write(s)
        



# fname = '/Users/devanshisingh/Desktop/FYP/text_files/content.txt'
# f = codecs.open(fname,'r', encoding='utf-8')
# text = f.read()
# divide_rephrase(text,fname)
# text = "Obama is the the president of US. Florida is a nice place. It is good. He lives in Florida. Trump is the current president. He owns Trump tower"

# print coref_rephrase(text)