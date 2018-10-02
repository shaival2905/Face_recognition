# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:24:58 2018

@author: Admin
"""

import cv2
import time
import os
from train_face import TrainFace
import imutils


class FaceSample:
    
    def __init__(self,detector, model):
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
    
    
    #takes five sample images for training    
    def take_sample(self, name, uniqueId, video=0):
    
        """
        arguments:
            name -> name of the person
            uniqueId -> unique identity of the device
            video -> default value 0 for webcam
        return:
            boolean value if face trained properly or not
        """
        
        detector = self.getDetector()
        
        
        video_capture = cv2.VideoCapture(video)
        sample_count=0
        if not os.path.isdir(uniqueId):
            os.mkdir(uniqueId)
        #images are stored in the train folder
        train = os.path.join(uniqueId,'train')
        if not os.path.isdir(train):
            os.mkdir(train)
           
        skip = 10
        pause = -1
        
        start_time = time.time()
        while(True):
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            pause+=1
            # skip frames
            if pause % skip != 0:
                continue
            
            # return if program is not able to take 5 samples until video stops
            if ret == False:
                return False
            
            if video == 0:
                frame = cv2.flip(frame, 1,0)
            else:
                frame = imutils.rotate(frame, 270)
            cv2.resize(frame,(350,350))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            image = detector(gray,1)
            
            if len(image) != 0 and len(image) == 1: #check whether one face is found in the frame
                image=image[0]
                
                sample_path = os.path.join(train,name)
                
                #creates respective folders for the images for each name
                if not os.path.isdir(sample_path):
                    os.mkdir(sample_path)
                cv2.imwrite(sample_path+'/'+str(sample_count)+'.jpg',frame)
                sample_count=sample_count+1
                if(sample_count==5):
                    video_capture.release()
                    break
            cv2.imshow('face',frame)
            cv2.waitKey(1)
        
        video_capture.release()
        cv2.destroyAllWindows()
        output = TrainFace(detector, self.getModel()).trainFace(name, uniqueId)
        time_elapsed = self.elapsed(time.time() - start_time) # calculate the time elapsed
        print("Elapsed time: ", time_elapsed)
        return output
	
