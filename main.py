import numpy
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
import os
import numpy as np
import tensorflow as tf
from yolo_utils import *


classNames = ['prohibitory','danger','mandatory','other']

# read the model cfg and weights with the cv2 DNN module
modelConfig_path = 'D:/final proj/yolo/weights/yolov3_training.cfg'
modelWeights_path = 'D:/final proj/yolo/weights/yolov3_training_last.weights'
neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)
# set the preferable Backend to GPU
neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
network = neural_net
height, width = 416,416

# confidence and non-max suppression threshold for this YoloV3 version
confidenceThreshold = 0.3
nmsThreshold = 0.6



root=Tk()
root.geometry('500x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Driver Cam')
frame.config(background='light blue')
label = Label(frame, text="Driver Cam",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="D:/final proj/yolo/images/demo.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)



def hel():
    help(cv2)

def Contri():
    tkinter.messagebox.showinfo("Contributors","\n1.Pritesh Gurjar\n2. Neev Shirke \n3. Malay Khakhar \n4. Navin Bubna \n")


def anotherWin():
    tkinter.messagebox.showinfo("About",'Driver Cam version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')
                                    
   

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV Docs",command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Driver Cam",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)



def exitt():
    exit()

  
def web():
    capture =cv2.VideoCapture(0)
    while True:
        ret,frame=capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

def webrec():
    capture =cv2.VideoCapture(0)
    fourcc=cv2.VideoWriter_fourcc(*'XVID') 
    op=cv2.VideoWriter('D:/final proj/yolo/output/Sample4.avi',fourcc,11.0,(640,480))
    while True:
        ret,frame=capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',frame)
        op.write(frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    op.release()
    capture.release()
    cv2.destroyAllWindows()   


def webdetRec():
    capture =cv2.VideoCapture(0)
    
    fourcc=cv2.VideoWriter_fourcc(*'XVID') 
    op=cv2.VideoWriter('D:/final proj/yolo/output/Sample4.avi',fourcc,11.0,(640,480))

   
    while True:

        ret, img = capture.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray)
        outputs = convert_to_blob(img, network, height, width)    

        bounding_boxes, class_objects, confidence_probs = object_detection(outputs, img, confidenceThreshold)  
        indices = nms_bbox(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold)

        box_drawing(img, indices, bounding_boxes, class_objects, confidence_probs, classNames, color=(	255, 255, 255), thickness=2)
    
        cv2.imshow('frame',img)
        op.write(img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    op.release()
    capture.release()
    cv2.destroyAllWindows()

################################ voice alert ########################

def danger():
    mixer.init()
    alert=mixer.Sound('D:/final proj/yolo/Audio/danger.wav')
    alert.play()
    time.sleep(0.3)
#    alert.play()  
def prohibitory():
    mixer.init()
    alert=mixer.Sound('D:/final proj/yolo/Audio/prohibitory.wav')
    alert.play()
    time.sleep(0.3)
#     alert.play()
def mandatory():
    mixer.init()
    alert=mixer.Sound('D:/final proj/yolo/Audio/mandatory.wav')
    alert.play()
    time.sleep(0.3)
#    alert.play()
def other():
    mixer.init()
    alert=mixer.Sound('D:/final proj/yolo/Audio/other.wav')
    alert.play()
    time.sleep(0.3)
#    alert.play()
   
def alert():
    mixer.init()
    alert=mixer.Sound('D:\Drowsiness-monitoring-Using-OpenCV-Python-master\Beep-07.wav')
    alert.play()
    time.sleep(0.1)
    alert.play()   
   
def blink():

    capture =cv2.VideoCapture(0)
   
    while True:

        ret, img = capture.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray)
        outputs = convert_to_blob(img, network, height, width)    

        bounding_boxes, class_objects, confidence_probs = object_detection(outputs, img, confidenceThreshold)  
        indices = nms_bbox(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold)

        temp , class_lbl=box_drawing(img, indices, bounding_boxes, class_objects, confidence_probs, classNames, color=(	255, 0, 255), thickness=2)

        class_lbl = np.array(class_lbl)
        class_lbl = np.unique(class_lbl)
        for i in class_lbl:
            print(i)
            if(i=="danger"):
                danger()
            elif(i=="prohibitory"):
                prohibitory()
            elif(i=="mandatory"):
                mandatory()
            else:
                other()
            
        cv2.imshow('frame',img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

   
but1=Button(frame,padx=5,pady=5,width=39,bg='black',fg='white',relief=GROOVE,command=web,text='Open Cam',font=('helvetica 15 bold'))
but1.place(x=5,y=104)

but2=Button(frame,padx=5,pady=5,width=39,bg='black',fg='white',relief=GROOVE,command=webrec,text='Open Cam & Record',font=('helvetica 15 bold'))
but2.place(x=5,y=176)

# but3=Button(frame,padx=5,pady=5,width=39,bg='black',fg='white',relief=GROOVE,command=webdet,text='Open Cam & Detect lables',font=('helvetica 15 bold'))
# but3.place(x=5,y=250)

but4=Button(frame,padx=5,pady=5,width=39,bg='black',fg='white',relief=GROOVE,command=webdetRec,text='Detect lables & signs Record',font=('helvetica 15 bold'))
but4.place(x=5,y=250)

but5=Button(frame,padx=5,pady=5,width=39,bg='black',fg='white',relief=GROOVE,command=blink,text='Detect Lables & sign With Sound',font=('helvetica 15 bold'))
but5.place(x=5,y=322)

but5=Button(frame,padx=5,pady=5,width=5,bg='black',fg='white',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but5.place(x=210,y=400)


root.mainloop()

