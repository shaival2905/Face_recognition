# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:26:39 2018

@author: shaival
"""


import cv2
import os
import numpy,shutil


class TrainDeleteLBPH:
    
    def __init__(self, model):
        self.model = model
        
    def getModel(self):
        return self.model
    
    def trainLBPH(self, name, idl, uniqueId):
        
        """
        arguments:
            name -> name of person to be trained. If value is none then train all faces else train this person only
            idl -> label to be assign to the person. Ignore if name = None
            uniqueId -> uniqueId of person to be trained
        
        """
        
        model = self.getModel()
        
        #creates a path for the model of each id
        trainedmodel = os.path.join(uniqueId, 'save_model.yaml')
        
        #face folder stores the gray images for training LBPH model
        face_dir = os.path.join(uniqueId, 'face')
            
        # collect images of only new person who is entered
        def collect_latest_images(name,id):
            
            #initialise images and labels
            (images, labels) = ([], [])
            # path for collecting images of entered person
            faces_path = os.path.join(face_dir, name)
            print(faces_path)
            if not os.path.isdir(faces_path):
                return [], []
            
            for filename in os.listdir(faces_path):
                
                # split file name and extension
                f_name, f_extension = os.path.splitext(filename)
                
                # check for wrong file type
                if(f_extension.lower() not in
                        ['.png','.jpg','.jpeg','.gif','.pgm']):
                    print("Skipping "+filename+", wrong file type")
                    continue
                
                # collect images one by one
                path = os.path.join(faces_path, filename) 
                print(path)
                label = id # assign label to face collected
                image = cv2.imread(path)
                image = cv2.resize(image,(250,250))
                # convert to gray scale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(gray)
                labels.append(int(label))
            
            # convert image and label into numpy array
            (images, labels) = [numpy.array(lis) for lis in [images, labels]]
            return images, labels
        
        # collect all the images which is stored in face folder
        def collect_all_images():
            
            flag = True # flag to check if any trained person remaining in the dataset 
            
            # initialise required variables
            (images, labels, id) = ([], [], 0)
            for (subdirs, dirs, files) in os.walk(face_dir):
                if len(dirs) == 0:
                    flag = False    
                for subdir in dirs:
                    face_path = os.path.join(face_dir, subdir)
                    print(face_path)
                    for filename in os.listdir(face_path):
                        
                        # split file name and extension
                        f_name, f_extension = os.path.splitext(filename)
                        
                        # check for wrong file type
                        if(f_extension.lower() not in
                                ['.png','.jpg','.jpeg','.gif','.pgm']):
                            print("Skipping "+filename+", wrong file type")
                            continue
                        
                        path = os.path.join(face_path, filename) 
                        print(path)
                        label = id
                        image = cv2.imread(path)
                        image = cv2.resize(image,(250,250))
    
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        images.append(gray)
                        labels.append(int(label))
                    id += 1
            
            (images, labels) = [numpy.array(lis) for lis in [images, labels]]
            return images, labels, flag
        
        if name == None:
            images, labels, flag = collect_all_images()
            
            # if no directory remaining delete previously trained model 
            if flag == True:
                for i in range(len(images)):
                    print(labels[i])
                model.train(images, labels)
                model.write(trainedmodel)
            else:
                os.remove(trainedmodel)
        else:
            images, labels = collect_latest_images(name,idl)
            
            if len(images) == 0 and len(labels) == 0:
                return False
            
            for i in range(len(images)):
                print(labels[i])
            if (os.path.isfile(trainedmodel)):
                model.read(trainedmodel)
                model.update(images, labels)
            else: 
                model.train(images, labels)
            model.write(trainedmodel)
            return True
    
    def delete_face_lbph(self, name, uniqueId):
        faces = os.path.join(uniqueId,'face')
        try:
            shutil.rmtree(faces+"/"+name) # remove directory of faces of seleted person
            self.trainLBPH(None, 0,uniqueId) # after deleting train all faces again
        except:
            print("Directory not there")
            return False
    
        return True
    

