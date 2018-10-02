# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:23:14 2018

@author: shaival
"""

import cv2, numpy, os, shutil
import sqlite3
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import numpy as np
import cluster


size = 1
fn_haar = 'haarcascades\\haarcascade_frontalface_default.xml'
trainedmodel = 'save_model.yaml'
fn_dir = 'att_faces'
in_dir = 'image'
(im_width, im_height) = (112, 92)
model=""

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)
    

def insert(Id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="insert into people(Id,Name) values("+str(Id)+", "+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
    
def nextid():
    conn=sqlite3.connect("FaceBase.db")
    cmd="select max(id) from people"
    cursor=conn.execute(cmd)
    ids=None
    for row in cursor:
        ids = row
    ids = ids[0]
    if(ids == None):
        return 1
    conn.close()
    return ids+1

global names
names = {}
names[-1] = "unknown"
def collect_latest_images(lid,subdir):
    images, labels, id = [], [], lid
    names[id] = subdir
    subjectpath = os.path.join(fn_dir, subdir)
    
    for filename in os.listdir(subjectpath):

        f_name, f_extension = os.path.splitext(filename)
        if(f_extension.lower() not in
           ['.png','.jpg','.jpeg','.gif','.pgm']):
            print("Skipping "+filename+", wrong file type")
            continue
        path = subjectpath + '/' + filename
        label = id

        images.append(cv2.imread(path, 0))
        labels.append(int(label))
    
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]
    return images, labels, names, id

def collect_all_images():
    (images, labels, id) = ([], [], 0)
    for (subdirs, dirs, files) in os.walk(fn_dir):
    
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(fn_dir, subdir)
    
            for filename in os.listdir(subjectpath):
    
                f_name, f_extension = os.path.splitext(filename)
                if(f_extension.lower() not in
                        ['.png','.jpg','.jpeg','.gif','.pgm']):
                    print("Skipping "+filename+", wrong file type")
                    continue
                path = subjectpath + '/' + filename
                label = id
    
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]
    return images, labels, names, id

def align_face(frame, gray, rect):
    faceAligned = fa.align(frame, gray, rect)
    faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
    faces1 = haar_cascade.detectMultiScale(faceAligned)
    if(len(faces1)==0):
        return faceAligned,faceAligned, False
    faces1 = faces1[0]
    (x1, y1, w1, h1) = [v for v in faces1]
    faceAligned = faceAligned[y1:y1 + h1, x1:x1 + w1]
    face_resize = cv2.resize(faceAligned, (im_width, im_height))
    return faceAligned, face_resize, True

def create_blank(width, height):
    rgb_color=(255,255,255)
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

blank_image = create_blank(500, 500)
blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
#j=0
j=[]

images, labels, names, id = collect_all_images()
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)

flag=False
flag1=False
f1=[]
known_faces=[]
face_label = -1
count = 0
#count = []
prediction = [-1,0]
tpred=[]
prev_prediction = []
for i in range(10):
        j.append(0)
        f1.append(True)

def reset():
    for i in range(10):
        prev_prediction.append(prediction)
        tpred.append(prediction)

reset()
pause = 0
count_max = 20
faceTrackers = {}
currentFaceID = 0
detector = dlib.get_frontal_face_detector()
model = cv2.face.LBPHFaceRecognizer_create(neighbors=10)
prediction = [-1,0]
def calc_threshold(height,width):
    area = height*width
    threshold = 120-(5.34/1000*area)
    if threshold < 70:
        return 70
    elif threshold > 110:
        return 110
    else: 
        return threshold

if (os.path.isfile(trainedmodel)):
    model.read("save_model.yaml")
    print(model)
    print("Model Loaded")
else:
    model.train(images, labels)
    #model.write(trainedmodel)
    print("Model trained and saved")
nf=0
while True:   
    rval = False
    if(not rval):
        (rval, frame) = webcam.read()
        if(not rval):
            cv2.destroyAllWindows()
            break        
    height, width, channels = frame.shape
    frame=cv2.flip(frame,1,0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
    
    fidsToDelete = []
    for fid in faceTrackers.keys():
        trackingQuality = faceTrackers[ fid ].update(gray)

        #If the tracking quality is good enough, we must delete
        #this tracker
        if trackingQuality < 7:
            fidsToDelete.append( fid )

    for fid in fidsToDelete:
        print("Removing fid " + str(fid) + " from list of trackers")
        faceTrackers.pop( fid , None )
    
    faces = haar_cascade.detectMultiScale(gray)
    faces = sorted(faces, key=lambda x: x[3], reverse=True)
    if(len(faces)!=nf):
        reset()
    nf = len(faces) 
    
    knf=0    
    for i in range(nf):
        face_i = faces[i]

        (x, y, w, h) = [v * size for v in face_i]
        threshold = calc_threshold(h,w)
        x_bar = x + 0.5 * w
        y_bar = y + 0.5 * h
        
        matchedFid = None

        #Now loop over all the trackers and check if the 
        #centerpoint of the face is within the box of a 
        #tracker
        for fid in faceTrackers.keys():
            tracked_position =  faceTrackers[fid].get_position()
            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            
            #calculate the centerpoint
            t_x_bar = t_x + 0.5 * t_w
            t_y_bar = t_y + 0.5 * t_h

            #check if the centerpoint of the face is within the 
            #rectangleof a tracker region. Also, the centerpoint
            #of the tracker region must be within the region 
            #detected as a face. If both of these conditions hold
            #we have a match
            if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                ( t_y <= y_bar   <= (t_y + t_h)) and 
                ( x   <= t_x_bar <= (x   + w  )) and 
                ( y   <= t_y_bar <= (y   + h  ))):
                            matchedFid = fid

        if matchedFid is None:
            count=0
            f1[i]=True
            j[i]=0
            print("in")
            """if(nf == 0 and count != 0):
                count=0
                idl = "id" + str(nextid())
                shutil.rmtree(fn_dir+"/"+idl, ignore_errors=True) """
            print("Creating new tracker " + str(currentFaceID))

            #Create and store the tracker 
            tracker = dlib.correlation_tracker()
            tracker.start_track(gray,
                                dlib.rectangle( x-10,
                                               y-20,
                                               x+w+10,
                                               y+h+20))
            faceTrackers[ currentFaceID ] = tracker
            currentFaceID += 1
         
            
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
       
        faceAligned, face_resize, rface = align_face(frame, gray, rect)
        if rface == False:
            continue
        

        j[i] = j[i] + 1
        if(tpred[i][1]<70):
            prev_prediction[i] = tpred[i]
        prediction = model.predict(faceAligned)
        tpred[i] = prediction
        if prediction[1]<round(threshold):
            
            cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            j[i]=0
            f1[i]=False
            knf = knf + 1
            (x1, y1, w1, h1) = (x, y, w, h)
            
            
        elif(j[i]>30):
            flag=True
            f1[i]=True
            
            print("j1"+str(i)+"=",j[i])
            if(count < count_max):
                if(w * 24 < width or h * 24 < height):
                    print("Face too small")
                else:   
                    if(pause == 0):
                        
                        path = in_dir
                        if knf > 0:
                            print("yes")
                            gray[y1:y1 + h1, x1:x1 + w1] = blank_image[0:0+h1, 0:0+w1]
                            
                        print("Saving training sample "+str(count+1)+"/"+str(count_max))
                        cv2.imwrite(path+"/"+str(count+1)+".jpg", gray)
                        count += 1
            
                        pause = 1
            
                if(pause > 0):
                    pause = (pause + 1) % 2
                    
            else:  
                print("j2=",j[i])
                id1 = nextid()
                if not os.path.isdir(in_dir):
                    os.mkdir(in_dir)
                num_classes = cluster.cluster(in_dir,fn_dir,id1)
               
                for k in range(num_classes):    
                    name = "id" + str(id1)
                    name1 = "\""+name +"\""
                    insert(id1,name1)
                    images, labels, names, id = collect_latest_images(id1,name)
                    model.update(images, labels)
                    id1 = id1 + 1
                f1[i]=True
                #model.write(trainedmodel)
                count=0
                j[i]=0
        
        else:
            if f1[i] == False:
                cv2.putText(frame,'%s - %.0f' % (names[prev_prediction[i][0]],prev_prediction[i][1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 
              
        if(flag==True):
            break
       
    
    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if key == 27:
        webcam.release()
        cv2.destroyAllWindows()
        break
print(names,labels)