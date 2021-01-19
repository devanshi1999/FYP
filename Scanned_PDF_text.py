import pandas as pd
import pytesseract as pt
import pdf2image

pages = pdf2image.convert_from_path(pdf_path='/Users/devanshisingh/Desktop/FYP/demo.pdf', dpi=200, size=(1654,2340))

for i in range(len(pages)):
    pages[i].save('demo' + str(i) + '.jpg')

content = pt.image_to_string(pages[0], lang='eng')
print(content)