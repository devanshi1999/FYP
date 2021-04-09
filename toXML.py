import xml.etree.ElementTree as etree
from xml.dom import minidom
import os

def prettify(elem):
    rough_string = etree.tostring(elem, 'utf-8', method='xml')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent='')

def createXML(q,A,fname):
    qid=1
    root = etree.Element("QApairs")
    root.set('id','1')
    
    ques = etree.SubElement(root,"question")
    ques.text = '\n'+q

    for a in A:
        ans = etree.SubElement(root,"positive")
        ans.text = '\n'+a

    with open (fname, "wb") as files : 
        files.write(prettify(root)) 

        
# q = "How many passengers does Amtrak serve annually ?"
# A = ["Amtrak	annually	serves	about	21	million	passengers	.",
#     "Schulz	said	Amtrak	prepared	for	the	new	guarantee	program	by	training	its	25,000	employees	''	to	take	personal	initiative	and	do	what	is	necessary	to	solve	guest	problems.	''",
#     "Currently	,	about	24,000	employees	work	for	Amtrak	,	according	to	spokesman	Steven	Taubenkibel	.",
#     ]


# if not os.path.exists(outdir):
#     os.makedirs(outdir)

