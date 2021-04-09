# -*- coding: utf-8 -*-
from Tkinter import *
from input import *

class botGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Q & A - 💬 Chat Bot")
        self.root.geometry("500x670")
        self.root.configure(background="LightPink4")
        self.root.resizable(width=FALSE, height=FALSE)

        self.chatWindow = Text(self.root, bd=3, width=70, foreground="black", bg="floral white", font = ("Arial",17))
        self.chatWindow.place(x=4,y=4, height=500, width=470)

        self.chatWindow.tag_configure('tag-left', justify='left', foreground = 'white', background='LightPink4', lmargin1 = 3)
        self.chatWindow.tag_configure('tag-right', justify='left', foreground = 'LightPink4', background='ivory2', lmargin1 = 3)
        self.chatWindow.tag_configure('break', justify='left', background='white')

        self.chatWindow.insert(END,"Hi! Ask me something.                                                           ",'tag-right')
        self.chatWindow.insert(END,"\n",'break')

        self.heading = Label(self.root, text="✏️ Your question here...", font = ("Arial",17))
        self.heading.configure(background='LightPink4',foreground = 'ivory2')
        self.heading.place(x=4,y=520)

        self.qWindow = Text(self.root, width=30, font = ("Arial",16), bd=4, background='floral white')
        self.qWindow.place(x=4, y=550, height=110, width=350)

        self.scrollbar = Scrollbar(self.root, command=self.chatWindow.yview, cursor="star", background='floral white')
        self.scrollbar.place(x=476,y=4, height=500)

        self.send = Button(self.root, text="▶ Send",  width=12, foreground='#ffffff', highlightbackground='LightPink4', command = self.get_question)
        self.send.place(x=360, y=570)
        self.back = Button(self.root, text="⬅ Back",  width=12, foreground='#ffffff', highlightbackground='LightPink4', command = self.go_back)
        self.back.place(x=360, y=600)

    def get_answer(self):
        # self.answer = new_question(self.question)
        # self.chatWindow.insert(END,"A: "+self.answer,'tag-right')
        self.chatWindow.insert(END,"A: ",'tag-right')
        self.chatWindow.insert(END,"\n",'break')
        
    def get_question(self):
        self.question = self.qWindow.get("1.0",END)
        self.chatWindow.insert(END,"Q: "+self.question,'tag-left')
        self.qWindow.delete("1.0",END)
        self.get_answer()
    
    def go_back(self):
        self.root.destroy()

    def start_bot(self):
        self.root.mainloop()

gui = botGUI()
gui.start_bot()