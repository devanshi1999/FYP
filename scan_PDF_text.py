import pandas as pd
import pytesseract as pt
import pdf2image
import sys
import re

def extract(path_to_pdf):
    pages = pdf2image.convert_from_path(pdf_path=path_to_pdf, dpi=200, size=(1654,2340))

    # saving content in a text file
    fname = '/Users/devanshisingh/Desktop/FYP/text_files/content.txt'
    f = open(fname,'w')
    for i,page in enumerate(pages):
        if i+5 == len(pages):
            break
        doc = pt.image_to_string(page, lang='eng')
        f.write(doc)

if len(sys.argv) > 1:
    path = sys.argv[1]
    print ("doc uploaded from", path)
    extract(path)
    print("PDF content saved in /text_files/content.txt")
else:
    print("No path entered")
# path = '/Users/devanshisingh/Downloads/iess301.pdf'

# 
# 
