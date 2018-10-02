# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:48:12 2018

@author: Admin
"""

import os
import cv2,shutil
from save_delete_encoding import SaveDeleteEncodings
from imutils.face_utils import rect_to_bb
from face_train_delete_lbph import TrainDeleteLBPH


class TrainFace:
    
    def __init__(self,detector, model):
        self.detector = detector
        self.model = model
    
    def getDetector(self):
        return self.detector
    
    def getModel(self):
        return self.model
    
    #trains the faces using LBPH
    def trainFace(self, name, uniqueId):
        
        detector = self.detector
        
        count_improper_faces=0 # counter for check improper faces
        # check if the required folder exist or make folders
        path_recognise = os.path.join(uniqueId,'face')
        if not os.path.isdir(path_recognise):
            os.mkdir(path_recognise)
        train_images = os.path.join(uniqueId,'train')
        if not os.path.isdir(train_images):
            os.mkdir(train_images)
    
        # initialize variables
        number_of_person = 0
        path = os.path.join(train_images, name)
        print(path)
        
        for filename in os.listdir(path):
            #get the file name and the extension    
            f_name, f_extension = os.path.splitext(filename)
            
            #accept files with the below mentioned extensions only
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Skipping "+filename+", wrong file type")
                continue
                
            image_path = os.path.join(path, filename) 
            image = cv2.imread(image_path)                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray,1) #detect faces in the frame
        
            if len(faces)!=1: #check whether more or less than one faces found
                continue
            
            face = faces[0]
            (x, y, w, h) = rect_to_bb(face)
            gray = gray[y:y+h,x:x+w]
            image = image[y:y+h,x:x+w]
            
            try:
                image = cv2.resize(image,(250,250))
                #save the encodings of the faces
                proper,number_of_person = SaveDeleteEncodings.save_encodings(image,f_name,name,uniqueId)
                print(proper) # to see encodings saved properly or not
                if proper == False:
                    count_improper_faces += 1
                # go to folder where gray scale images are saved
                path2 = os.path.join(path_recognise, name)
                if not os.path.isdir(path2):
                    os.mkdir(path2)
                cv2.imwrite(path2+"/"+f_name+".png",gray)
            except:
                print("Image not proper")
                        
            #once the images are trained, delete the images
        try:
            shutil.rmtree(path) 
        except:
            print("Directory not there")
    
        # return false if more than 2 encodings are not proper        
        if count_improper_faces > 2:
            return False        
                    
        #train the faces using LBPH  
        if not TrainDeleteLBPH(self.getModel()).trainLBPH(name,number_of_person-1,uniqueId):
            SaveDeleteEncodings.delete_encodings(name, uniqueId) # method to remove encodings from saved encodings
            return False
        
        return True
     