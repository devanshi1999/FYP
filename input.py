from text_preprocessing import *
from idf_bm25 import *
from toXML import *
import subprocess
import codecs

def new_doc_upload(docpath):
    subprocess.call(['sh','run.sh', docpath])
    # print docpath

def run_on_NN():
    subprocess.call(['sh','find_ans.sh'])
    print "prediction results saved in /text_files/results.txt"

def new_question(question):
    # question = "When was Napoleon Bonaparte crowned emperor of France ?"
    query = preprocess_question(question)
    print "User Query Preprocessing Finished----"
    
    fname = '/Users/devanshisingh/Desktop/FYP/text_files/preprocessed.txt'
    f = codecs.open(fname,'r', encoding='utf-8')
    lines = f.readlines()

    doc_freq, N = find_doc_freq(lines)
    idf = find_idf(doc_freq,N)
    word_freq = find_word_freq(lines)
    doc_len, avg_len = find_doc_len(lines)
    doc_score = find_doc_score(query,word_freq,idf,doc_len,avg_len)

    kbname = '/Users/devanshisingh/Desktop/FYP/text_files/knowledgeBase.txt'
    f = codecs.open(fname,'r', encoding='utf-8')
    lines = f.readlines()

    print "Found top 10 answers based on BM25 score"
    topK_docid, topK_answers = topKanswers(lines,doc_score,10)

    outdir = '/Users/devanshisingh/Desktop/FYP/text_files/Jacana'
    createXML(question + '\n', topK_answers, outdir+"/test.xml")
    print "Jacana formatted XML file saved in /text_files/Jacana/test.xml"

    print "Running on neural network now for reranking"
    run_on_NN()

    fname = '/Users/devanshisingh/Desktop/FYP/text_files/ans_sentence.txt'
    f = codecs.open(fname,'r')
    lines = f.readlines()
    return lines[-1]


# if __name__ == '__main__':
#     # new_doc_upload('/Users/devanshisingh/Downloads/iess301.pdf')
    # print new_question("When did Napoleon Bonaparte become emperor of France ?")
#     print "main"