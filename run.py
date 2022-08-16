import cv2
import torch
from deep_emotion import Deep_Emotion
import numpy as np
import torch.nn.functional as F
import rtmidi
import time
from multiprocessing import Process, Pipe
import multiprocessing


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=Deep_Emotion()
net.load_state_dict(torch.load("deep_emotion-100-128-0.005.pt"))
net.to(device)

from curses.textpad import rectangle
path='haircascade_frontalface_default.xml'
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN

rectangle_bgr=(255,255,255)
img=np.zeros((500,500))

text = "Some text in a box!"
(text_width,text_height)=cv2.getTextSize(text,font,fontScale=font_scale, thickness=1)[0]
text_offset_x=10
text_offset_y=img.shape[0] - 25

box_coords= ((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y- text_height-2))
cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text, (text_offset_x,text_offset_y),font,fontScale=font_scale,color=(0,0,0),thickness=1)
cap=cv2.VideoCapture(1)

if cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open Webcam')


ret,frame = cap.read()
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)




x1,y1,w1,h1=0,0,174,75

#temp = tempfile.TemporaryFile(mode="w+b")
x=1

def exec(connection):
    print("Sender: Running", flush=True)
  
    while True:

        ret,frame = cap.read()
        faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,1.1,4)

        def roi():
              for x,y,w,h in faces:
                  roi_gray=gray[y:y+h,x:x+w]
                  roi_color=frame[y:y+h,x:x+w]
                  cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                  facess=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(roi_gray)
                  if len(faces)==0:
                      print("Face not detected")
                      return 0
                  else:
                      for (ex,ey,ew,eh) in facess:
                          face_roi = roi_color[ey: ey+eh,ex:ex+ ew]
                          return face_roi
                          
        if roi() is not None:
            graytemp= cv2.cvtColor(roi(), cv2.COLOR_BGR2GRAY)
            final_image=cv2.resize(graytemp, (48,48))
            final_image = np.expand_dims(final_image,axis=0)
            final_image = np.expand_dims(final_image,axis=0)
            final_image=final_image/255.0
            dataa=torch.from_numpy(final_image)
            dataa=dataa.type(torch.FloatTensor)
            dataa=dataa.to(device)
            outputs=net(dataa)

            Pred=F.softmax(outputs,dim=1)
            Predictions= torch.argmax(Pred)

            x=int(127*abs(Pred[0,3]))
    
            connection.send(x)
            #temp.write("{x}".encode("utf-8"))

            font_scale = 1.5
            font=cv2.FONT_HERSHEY_PLAIN

            if ((Predictions)==0):
               status = "Angry"
               cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
               cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
               
            elif ((Predictions)==1):
               status = "Disgust"
               cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
               cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

            elif ((Predictions)==2):
               status = "Fear"
               cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
               cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

            elif ((Predictions)==3):
               status = "Happy"
               cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
               cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

            elif ((Predictions)==4):
               status = "Sad"
               cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
               cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

            elif ((Predictions)==5):
               status = "Surprise"
               cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
               cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

            elif ((Predictions)==6):
               status = "Neutral"
               cv2.rectangle(frame, (x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
               cv2.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

            cv2.imshow('Face Emotion Recognition',frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
               break



        else:
           cv2.imshow('Face Emotion Recognition',frame)
           connection.send(0)
           print("oh no")
    
    connection.send(None)
    print("Sender: Done", flush=True)
       

def m1(connection):
    print("processing..", flush=True)
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    midiout.open_port(4)
    
    while True:
        #if ((time.time()-y)>0.001):
        item = connection.recv()
        #print(item)
        midiout.send_message([0x90, item, 127])
        #p = Process(target=m1, args=(xdat,lock))
        #p.start()
        time.sleep(0.0001)
        #y=time.time()
        if item is None:
            print("No messages recieved.")
        
    del midiout
    print("Receiver: Done", flush=True)
    #cc= [0xB0, 37, x]
    #cc = [0x25, 60, 112] # channel 1, middle C, velocity 112
    #midiout.send_message(note_on)
    #print(x,int(127*abs(Pred[0,5])),int(127*abs(Pred[0,6])))
    #p.terminate()

if __name__ == '__main__':


    conn1, conn2 = multiprocessing.Pipe(duplex=True)
    p1=Process(target=exec, args=(conn1,))
    p1.start()

    p2=Process(target=m1, args=(conn2,))
    p2.start()

    p1.join()
    p2.join()
    
    cap.release()
    cv2.destroyAllWindows()





