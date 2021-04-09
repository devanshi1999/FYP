from textblob import TextBlob 
import math
import codecs
from text_preprocessing import preprocess_question
import collections

def find_doc_freq(lines):   
    freq = {}

    N = len(lines)

    for line in lines:
        temp = {}                           
        bobj = TextBlob(line)
        words = bobj.words
        for word in words:
            word = str(word)
            freq.setdefault(word,0)
            temp.setdefault(word,0)
            temp[word] = 1
        
        for key in temp.keys():
            if temp[key] == 1:
                freq[key] = freq[key] + 1

    # print sorted(freq.items(), key=lambda x: (x[1]))

    return freq, N

def find_idf(freq, N):
    idf = {}
    for word in freq.keys():
        idf[word] = math.log((N - freq[word] + 0.5)/(freq[word] + 0.5))
    return idf

def find_word_freq(lines):
    freq = dict()

    for i,line in enumerate(lines):
        bobj = TextBlob(line)
        words = bobj.words
        for word in words:
            word = str(word)
            if word in freq:
                if i in freq[word]:
                    freq[word][i]+=1
                else:
                    freq[word][i]=1
            else:
                d=dict()
                d[i]=1
                freq[word]=d

    return freq

def find_doc_len(lines):
    doc_len = dict()
    avg_len = 0
    for i, line in enumerate(lines):
        bobj = TextBlob(line)
        words = bobj.words
        if i not in doc_len:
            doc_len[i] = len(words)
            avg_len += doc_len[i]
    
    avg_len = float(avg_len)/float(len(lines))

    return doc_len, avg_len

def BM25_score(word_idf,freq,docLen,avgDocLen):
    k = 1.2
    b = 0.75
    denom = (k * (1 - b + b * (float(docLen)/float(avgDocLen)))) + freq
    numer = word_idf * (freq * (k + 1))
    return numer/denom

def find_doc_score(query, word_freq, idf, doc_len, avg_len):
    doc_score = dict()
    words = query.split(' ')
    for word in words:
        if word in word_freq:
            for docid in word_freq[word]:
                score = BM25_score(idf[word], word_freq[word][docid], doc_len[docid], avg_len)
                if docid in doc_score:
                    doc_score[docid]+=score
                else:
                    doc_score[docid]=score
    # print doc_score
    return doc_score

def topKanswers(lines, doc_score, k):
    sorted_doc_score = sorted(doc_score.items(), key=lambda item: item[1], reverse=True)
    # print sorted_doc_score

    topK = sorted_doc_score[:k]
    # print topK

    d = collections.OrderedDict()
    
    for item in topK:
        d[item[0]] = item[1]

    answers = []

    for i,line in enumerate(lines):
        if i in d.keys():
            answers.append(str(line))
            # print i, str(line)

    return d, answers



# fname = 'preprocessed.txt'
# fname = 'demo.txt'
# f = codecs.open(fname,'r', encoding='utf-8')
# lines = f.readlines()

# doc_freq, N = find_doc_freq(lines)

# idf = find_idf(doc_freq,N)

# word_freq = find_word_freq(lines)

# doc_len, avg_len = find_doc_len(lines)




# doc_score = find_doc_score(query,word_freq,idf,doc_len,avg_len)

# topK = topKanswers(doc_score,10)

