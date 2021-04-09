# -*- coding: utf-8 -*-
from textblob import TextBlob 
from textblob import Word
import codecs
import string
import re
from nltk.stem import PorterStemmer 

def tokenize_to_sentences(blob_object):
    sentences = blob_object.sentences
    return sentences

def tokenize_to_words(blob_object):
    words = blob_object.words
    return words

def remove_misspelling(blob_object):
    correct_txt = blob_object.correct()
    return correct_txt

def remove_irrelevant_sentences(sentences):
    # sentences with word count<5 or are an url
    irrelevant = []
    for sentence in sentences:
        res1 = len(re.findall(r'\w+', str(sentence))) 
        res2 = len(re.findall("(www|https|http|Fig.)[a-z,0-9]*(.com)*",str(sentence)))
        if res1<=5 or res2>0:
            irrelevant.append(sentence)
    for s in irrelevant:
        sentences.remove(s)
    return sentences

def remove_punctuations(text):
    new_text = re.sub(r'[^\w\s]', '', text)
    return new_text

def singularize(blob_object):
    return blob_object.singularize()

def lemmatize(blob_object):
    return blob_object.lemmatize()

def stemming(text):
    ps = PorterStemmer() 
    return ps.stem(text)

def createKB(sentences,fname):
    f = open(fname,'w')
    for sentence in sentences:
        s = str(sentence) + '\n'
        f.write(s)
        
def preprocess_answers(blob_object):
    blob_object = remove_misspelling(blob_object)
    sentences = tokenize_to_sentences(blob_object)
    sentences = remove_irrelevant_sentences(sentences)
    # print sentences
    createKB(sentences,kbname)

    print "Knowledge Base created, saved in /text_files/knowledgeBase.txt"

    fname = '/Users/devanshisingh/Desktop/FYP/text_files/preprocessed.txt'
    fw = open(fname,'w')
    
    for sentence in sentences:
        s = remove_punctuations(str(sentence))
        bobj = TextBlob(s)
        bobj = remove_misspelling(bobj)
        words = tokenize_to_words(bobj)
        s = ""
        for word in words:
            word = singularize(word)
            word = lemmatize(word)
            # word = stemming(str(word))
            word = remove_misspelling(Word(word))
            word = str(word)
            word = word.lower()
            if len(word)==1 and word.isalpha():
                continue
            
            s=s+word+' '
        s = s + '\n'
        
        fw.write(s)
    print "Preprocessed text saved in /text_files/preprocessed.txt"

def preprocess_question(question):
    blob_object = TextBlob(question)
    s = remove_punctuations(str(blob_object))
    bobj = TextBlob(s)
    bobj = remove_misspelling(bobj)
    words = tokenize_to_words(bobj)
    
    s=""
    for word in words:
            word = singularize(word)
            word = lemmatize(word)
            # word = stemming(str(word))
            word = remove_misspelling(Word(word))
            word = str(word)
            word = word.lower()
            if len(word)==1 and word.isalpha():
                continue
            s=s+word+' '
    return s+'\n'

kbname = '/Users/devanshisingh/Desktop/FYP/text_files/knowledgeBase.txt'

question = "When did Napoleon, Bonaparte become king of France's?"
# blob_object_q = TextBlob(question)

# preprocess_question(blob_object_q)

# fname = './text_files/content.txt'
# f = codecs.open(fname,'r', encoding='utf-8')
# content = f.read()
# print content
# blob_object_a = TextBlob(content) 

# preprocess_answers(blob_object_a)
