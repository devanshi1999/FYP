from text_cleaning import *
from text_preprocessing import *
from rephrase import *
import re

def clean(content):
    content = remove_accented_chars(content)
    content = expand_contractions(content)
    content = remove_irrelevant_words(content)
    content = remove_special_characters(content)
    content = remove_extra_spaces_tabs(content)

    return content

def preprocess(content):
    blob_object_a = TextBlob(content) 
    preprocess_answers(blob_object_a)

def coreference(content,fname):
    divide_rephrase(content,fname)


def main():
    fname = '/Users/devanshisingh/Desktop/FYP/text_files/content.txt'
    f = codecs.open(fname,'r', encoding='utf-8')

    content = f.read()
    
    print "Text Cleaning Started---------------"
    clean_content = clean(content)
    clean_content = remove_irrelevant_words(clean_content)
    # f = open(fname,'w')
    # f.write(clean_content)
    print "Text Cleaning Finished---------------"

    print "Coreference Resolution Started---------------"
    coreference(clean_content,fname)
    print "Coreference Resolution done! saved in /text_files/content.txt---------------"

    print "Text Preprocessing Started---------------"
    f = codecs.open(fname,'r', encoding='utf-8')
    clean_content = f.read()
    clean_content = re.sub("(\s)(\.)",". ",clean_content)
    clean_content = re.sub("(\s)(\?)","? ",clean_content)
    clean_content = re.sub("(\s)(\!)","! ",clean_content)
    remove_extra_spaces_tabs(clean_content)
    preprocessed_answers = preprocess(clean_content)
    print "Text Preprocessing Finished---------------"
    

if __name__ == '__main__':
    main()


