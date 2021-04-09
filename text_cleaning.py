import unicodedata
from contractions import CONTRACTION_MAP 
import re
import nltk
import codecs

def remove_extra_spaces_tabs(text):
    new_text = re.sub("\s+"," ", text)
    return new_text

def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text):
    pat = r'[^a-zA-Z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)

def remove_irrelevant_words(text):
    new_text = re.sub("(SECTION|CHAPTER|Section|Chapter)[\s|\t]*[0-9]+",'',text)
    new_text = re.sub("(SECTION|CHAPTER|Section|Chapter)[\s|\t]*[I,V,X,L,C,D,M]+",'',new_text)
    new_text = re.sub("(SECTION|CHAPTER|Section|Chapter)",'',new_text)
    new_text = re.sub("(Diagram|Figure|Table|Exercises|Fig.)[\s|\t]*[0-9]+[.]*[0-9]*[.]*",'',new_text)
    new_text = re.sub("[0-9]+(\s)[0-9]+",'',new_text)
    # new_text = re.sub("(\s)('.')",". ",new_text)
    temp = new_text
    while True:
        new_text = re.sub("[\s][a-z,A-Z][\s]",' ',temp)
        if temp == new_text:
            break
        temp = new_text
    return new_text

