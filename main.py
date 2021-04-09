# -*- coding: utf-8 -*-
from  Tkinter import *
import tkMessageBox
from input import *
from tkinter import filedialog
from bot_gui import *

class GUI:
    def __init__(self,master):
        master.title("Q & A - File Upload")
        master.geometry("500x400")
        master.configure(background='misty rose')
        master.resizable(width=FALSE, height=FALSE)
        self.filename=''
        self.kbcreated=0

        self.l1 = Label(master, text="Upload a PDF Document", font = ("Arial Bold",20))
        self.l2 = Label(master, text="No file Selected")
        self.browse = Button(master, text ="🔎 Browse",command=self.fileDialog, font = ('Arial', 15))
        self.submit = Button(master, text ="✔️ Submit",command=self.upload, font = ('Arial', 15))
        self.bot = Button(master, text ="🗨️ Goto Chatbot",command=self.open_bot, font = ('Arial', 15))
        self.info = Button(master, text ="❓ About",command=self.show_info, font = ('Arial', 15))
		
        self.l1.configure(foreground='LightPink4',background='misty rose')
        self.l2.configure(foreground='slate gray',background='misty rose')
        self.browse.configure(width = 15,highlightbackground='misty rose')
        self.submit.configure(width = 15,highlightbackground='misty rose')
        self.bot.configure(width = 15,highlightbackground='misty rose')
        self.info.configure(width = 15,highlightbackground='misty rose')
		
        self.l1.place(x=135,y=30)
        self.l2.place(x=40,y=110, width=400)
        self.browse.place(x=170, y=180)
        self.submit.place(x=170, y=210)
        self.bot.place(x=170, y=240)
        self.info.place(x=170, y=270)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetypes = (("pdf files","*.pdf"),("all files","*.*")) )
        if self.filename:
            self.l2.configure(text = "File chosen: "+self.filename)
            
    def upload(self):
        if self.filename:
            new_doc_upload(self.filename)
            tkMessageBox.showinfo('Success!', 'File processed and knowledge base created.')
            self.kbcreated=1
        else:
            tkMessageBox.showinfo('oops!', 'You need to select a file!')

    def open_bot(self):
        if self.kbcreated == 1:
            bot_gui = botGUI()
            bot_gui.start_bot()
        else:
            tkMessageBox.showinfo('oops!', 'You need to first select a file and click submit!')

    def show_info(self):
        root = Tk()
        root.title("Q & A - About")
        root.geometry("450x500")
        root.resizable(width=FALSE, height=FALSE)
        root.configure(background='RosyBrown4')

        self.about = Label(root, text="A python chatbot that uses power of \nBM25, CNN & word vectors to \nfind best answer. Network is trained \nusing TREC QA Dataset.", foreground="misty rose", background="RosyBrown4", font = ("Courier New",16))
        self.about.place(x=35,y=20)

        self.metric = Label(root, text="Neural Network Metrics",font = ("Arial Bold",16), background="misty rose")
        self.metric.configure(width=56,height = 1, foreground="RosyBrown4")
        self.metric.place(x=0,y=110)

        self.l3 = Label(root, text="▫️ MRR: 0.7101\t\t▫️ MAP: 0.7708",font = ("Helvetica",17), foreground="misty rose", background="RosyBrown4")
        self.l3.place(x=35,y=140)

        self.use = Label(root, text="How to use ?",font = ("Arial Bold",16), background="misty rose")
        self.use.configure(width=56,height = 1, foreground="RosyBrown4")
        self.use.place(x=0,y=180)

        self.l4 = Label(root, text="1️⃣ Upload a PDF file to choose answers from.",font = ("Helvetica",16), foreground="misty rose", background="RosyBrown4", anchor=W)
        self.l5 = Label(root, text="2️⃣ Submit, to create knowldege base.",font = ("Helvetica",16), foreground="misty rose", background="RosyBrown4", anchor=W)
        self.l6 = Label(root, text="3️⃣ Open chatbot and start asking!",font = ("Helvetica",16), foreground="misty rose", background="RosyBrown4", anchor=W)

        self.l4.place(x=30,y=210)
        self.l5.place(x=30,y=235)
        self.l6.place(x=30,y=260)

        self.bg = Label(root, text="What happens in the background ?",font = ("Arial Bold",16), background="misty rose")
        self.bg.configure(width=56,height = 1, foreground="RosyBrown4")
        self.bg.place(x=0,y=300)

        self.l7 = Label(root, text="1️⃣ Text extraction from PDF.",font = ("Helvetica",16), foreground="misty rose", background="RosyBrown4", anchor=W)
        self.l8 = Label(root, text="2️⃣ Text cleaning, preprocessing and coreference resolution.",font = ("Helvetica",16), foreground="misty rose", background="RosyBrown4", anchor=W)
        self.l9 = Label(root, text="3️⃣ Candidate answer pool using BM25 similarity score.",font = ("Helvetica",16), foreground="misty rose", background="RosyBrown4", anchor=W)
        self.l10 = Label(root, text="4️⃣ Creating vocabulary & extracting the word vectors.",font = ("Helvetica",16), foreground="misty rose", background="RosyBrown4", anchor=W)
        self.l11 = Label(root, text="5️⃣ Running on NN and displaying best answer.",font = ("Helvetica",16), foreground="misty rose", background="RosyBrown4", anchor=W)

        self.l7.place(x=30,y=335)
        self.l8.place(x=30,y=360)
        self.l9.place(x=30,y=385)
        self.l10.place(x=30,y=410)
        self.l11.place(x=30,y=435)


        root.mainloop()

window = Tk()

gui = GUI(window)

window.mainloop()

