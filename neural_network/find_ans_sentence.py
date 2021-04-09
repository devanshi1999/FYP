import re
import subprocess

def load_data(fname):
    lines = open(fname).readlines()
    qids, questions, answers, labels = [], [], [], []
    num_skipped = 0
    previous_line = ''

    qid2num_answers = {}  #to store number of answers for a question or qid
    qid_question = {}   # to store Question associated with each qid
    qid_answers = {}   # to store Answer associated with each qid

    for i, line in enumerate(lines):
        line = line.strip()

        qid_match = re.match('<QApairs id=[\'|\"](.*)[\'|\"]>', line)

        # getting qid
        if qid_match:
            qid = qid_match.group(1)
            qid2num_answers[qid] = 0
            qid_question[qid] = ""
            qid_answers[qid] = []

        #getting question
        if previous_line and previous_line.startswith('<question>'):
            question = line.replace('\t',' ')
            qid_question[qid]=question

        #getting answers and labels viz positive - 1, negative - 0
        label = re.match('<(positive|negative)>', previous_line)
        if label:
            label = label.group(1)
            label = 1 if label == 'positive' else 0
            answer = line.replace('\t',' ')
            qid_answers[qid].append(answer)
            qid2num_answers[qid] += 1
        previous_line = line
    
    return qid2num_answers, qid_question, qid_answers 

def convertToDict(fname):
    lines = open(fname).readlines()
    qid_preds = {}
    qid_pred = 0
    qid_line = 0

    for i, line in enumerate(lines):
        line = line.strip()
        row = line.split(' ')
        qid = row[0]
        if qid in qid_preds.keys():
            # key already present
            if row[1] > qid_pred:
                qid_pred = row[1]
                qid_preds[qid] = i - qid_line
        else:
            # insert key
            qid_line = i
            qid_pred = row[1]
            qid_preds[qid] = 0

    return qid_preds
        
def find_sentence(qid_question, qid_answers, qid_preds, fname):
    f = open(fname,'w')
    for qid in sorted(qid_preds.keys()):
        print "Q:", qid_question[qid]
        print "A:", qid_answers[qid][qid_preds[qid]]
        f.write(qid_question[qid])
        f.write("\n")
        f.write(qid_answers[qid][qid_preds[qid]])
        
# Jacana parsed file which has all qids and question and answer pairs
fname = "/Users/devanshisingh/Desktop/FYP/text_files/test.txt"

qid2num_answers, qid_question, qid_answers  = load_data(fname)

# file which has all prediction probabilities
fname = '/Users/devanshisingh/Desktop/FYP/text_files/results.txt'

qid_preds = convertToDict(fname)

fname = '/Users/devanshisingh/Desktop/FYP/text_files/ans_sentence.txt'

find_sentence(qid_question, qid_answers, qid_preds, fname)
