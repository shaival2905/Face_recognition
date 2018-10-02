# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:07:43 2018

@author: shaival
"""

import face_recognition
import pickle
import numpy as np
import cv2
from imutils.face_utils import rect_to_bb
import time
import os

class Recognize:
    
    def __init__(self,detector,model):
        self.detector = detector
        self.model = model
    
    def getDetector(self):
        return self.detector
    
    def getModel(self):
        return self.model
    
    # calculate the elapsed time
    def elapsed(self,sec):
        if sec<60:
            return str(sec) + " sec"
        elif sec<(60*60):
            return str(sec/60) + " min"
        else:
            return str(sec/3600) + " hr"
    
    
    """
    
    function for auto rotate frame 
    
    
    """
    
    # reognize face: returns known or unknown        
    def recognize_face(self, uniqueId, video=0):
        
        """
        arguments:
            uniqueId -> uniqueId of a person which is to be recognised
            video -> default value 0 for webcam else to run on ip camera or on video
        return:
            string -> returns person is known or unknown
        """
        
        detector = self.getDetector()
        model = self.getModel()
    
    
        trainedmodel = os.path.join(uniqueId, 'save_model.yaml') #path to lbph trained model
        frame_to_check=3 # take 3 frames to process output, so results will be collected from 3 frames
        
        #check if the trained model exist ot not
        if os.path.isdir(uniqueId) and (os.path.isfile(trainedmodel)):
            model.read(trainedmodel)
            print("Model Loaded")
            model_found=True
        else:
            model_found=False
        # check if any encodings exist or not
        try:
            dataset = os.path.join(uniqueId, 'dataset_faces.dat')
            with open(dataset, 'rb') as f:
                all_face_encodings = pickle.load(f)
            stop = False
        except:
            print("Encodings not found")
            stop = True
        
        # process if model and encodings found 
        if stop == False and model_found == True:
            
            video_capture = cv2.VideoCapture(video) # run video or webcam
            
            if video !=0:
                fps = video_capture.get(cv2.CAP_PROP_FPS)
            else:
                fps = 20
            # initialise variables
            face_known=False
            kname='Unknown'
            ukfcount=0
            kfcount=0
            kencode={}
            lbph_count=0
            
            for names in all_face_encodings:
                kencode[names]=0
            
            count_not_proper_frame=0   
            start_time = time.time()
            
            while True:

                ret, frame = video_capture.read() # read frame
                
                # if frame not found skip the processing
                if not ret:
                    continue
                
                cv2.imshow('opencv',frame)
                #continuously no is looking at the camera or not proper frame then return
                if count_not_proper_frame > fps*2:
                    return 'notProper', None
             
                
                count_not_proper_frame+=1
                
                if video == 0:
                    frame = cv2.flip(frame, 1,0)# flip the frame
                
                #else:
                    #frame = imutils.rotate(frame, 270)
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert into gray scale
                faces = detector(gray,1) # fetch faces in the frame
                
                # if more than 1 face found do not process 
                if len(faces)!=0:
                    print('Please look at the front of camera')
                    continue
                
                
                try:
                    face = faces[0] # get dlib rectangle 
                    (x, y, w, h) = rect_to_bb(face) # get coordinates from dlib rectangle 
                    image = frame[y:y+h,x:x+w] # crop face from image
                    face = gray[y:y+h,x:x+w] # crop face from gray scale image
                    face = cv2.resize(face,(250,250)) # resize image
                except:
                    continue
                    
                #reset the counter after checking if video is proper or not
                count_not_proper_frame=0
                
                # predict face using lbph model
                prediction = model.predict(face)
                if prediction[1]<65:
                    lbph_count = lbph_count + 1
                    print("LBPH: True")
                else:
                    print("LBPH: False")
                
                for names in all_face_encodings:
    
                    # Grab the list of names and the list of encodings
                    face_names = list(all_face_encodings[names].keys())
                    
                    # take encodings of person at one time
                    face_encodings = np.array(list(all_face_encodings[names].values()))
                    
                    # Try comparing an unknown image
                    unknown_face = face_recognition.face_encodings(image)
                    shape = np.array(unknown_face).shape
                    if len(shape) != 2 or shape[1] != 128:
                        break
                    result = face_recognition.compare_faces(face_encodings, unknown_face, tolerance=0.5)
                    
                    # initialize variables to check how much time algorithms is returning true of false out of encodings saved
                    kface=0
                    ukface=0
                    
                    # take the result as a list of names with True/False
                    names_with_result = list(zip(face_names, result))
                    for ans in names_with_result:
                        if ans[1] == False:
                            ukface+=1
                        else:
                            kface+=1
                    
                    # Find the majority vote for the matches and unmatches
                    if kface > ukface:
                        face_known=True
                        kencode[names] = kencode[names] + kface
                        kname = names
                        break
                    else:
                        face_known=False
                    
                
                # if face matches in current frame increase count of known face else increase count of unknown face
                if face_known == True:
                    kfcount+=1
                else:
                    ukfcount+=1
                
                # if either of the both count values reaches to maximum number of frames to be process return the result
                if kfcount==frame_to_check:
                    print('DLIB: True')
                    video_capture.release()
                    break
                if ukfcount==frame_to_check:
                    print('DLIB: False')
                    video_capture.release()
                    break
            
            # Taking dlib as primary algorithm and lbph as secondary get final results
            for i in kencode:
                if kencode[i] == 5*frame_to_check or kencode[i] >= 5*frame_to_check-2:
                    final_result = True
                    break
                elif kencode[i]>5*frame_to_check-6 and lbph_count>=frame_to_check-2:
                    final_result = True
                    break
                elif kencode[i]>5*frame_to_check-7 and lbph_count >= frame_to_check-1:
                    final_result = True
                    break
                else:
                    final_result = False
            
            time_elapsed = self.elapsed(time.time() - start_time) # calculate the time elapsed
            print("Person name: ",kname)        
            print('Final result: ',final_result)
            print("Elapsed time: ", time_elapsed)
            
            # return final result
            if final_result == True:
                return 'known', kname
            else:
                return 'unknown', None
            
        else:
            return 'false', None # return is either model or encodings are not found for the particular id


    