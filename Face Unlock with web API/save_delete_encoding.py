# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 21:50:56 2018

@author: Admin
"""

import face_recognition
import pickle
import os
from collections import defaultdict

class SaveDeleteEncodings:
    
    # save encodings of person 
    # open previously stored encodings and append new encodings in it and store it again
    def save_encodings(image, orientation, name, uniqueId):
        
        """
        arguments:
            image -> face captured from frame
            orientation -> orientation of face or image name
            name -> name of person whose encodings are to be saved
            uniqueId -> uniqueId of a person which is to be trained
        
        return:
            boolean value -> to check if encodings saved properly or not
            length of total encodings saved -> to check total number of persons stored in dataset
        """
        
        all_face_encodings = defaultdict(dict) # 2D dictionary for storing person and his 5 encodings
        path = os.path.join(uniqueId,'dataset_faces.dat') # moved to uniqueId folder
        
        # load the saved encoding files
        if (os.path.isfile(path)):
            with open(path, 'rb') as f:
                all_face_encodings = pickle.load(f)
        try:
            all_face_encodings[name][orientation] = face_recognition.face_encodings(image)[0] # store encodings in dictionary
        except:
            print("Face is not proper") # if not able to get encodings properly
            return False, len(all_face_encodings)
        
        # save updated encodings
        with open(path, 'wb') as f:
            pickle.dump(all_face_encodings, f)
        
        return True, len(all_face_encodings)
    
    
    # open stored encodings and delete encodings of person which is entered and store again
    def delete_encodings(name, uniqueId):
        
        """
        arguments:
            name -> name of person whose encodings are to be deleted
            uniqueId -> uniqueId of a person which is to be deleted
        
        return:
            boolean value -> to check if encodings deleted or not
            length of total encodings saved -> to check total number of persons remaining in dataset
        """
        
        all_face_encodings = defaultdict(dict) # 2D dictionary for storing person and his 5 encodings
        path = os.path.join(uniqueId,'dataset_faces.dat') # moved to uniqueId folder
        
        # load the saved encoding files
        if (os.path.isfile(path)):
            with open(path, 'rb') as f:
                all_face_encodings = pickle.load(f)   
        try:
            del all_face_encodings[name] # delete encodings of person
            
            #saved updated encodings
            with open(path, 'wb') as f:
                pickle.dump(all_face_encodings, f)
            
            return True, len(all_face_encodings)
        
        except:
            print(name+"'s face is not saved")
        
        return False, len(all_face_encodings)