# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 12:19:24 2018

@author: shaival
"""

from face_train_delete_lbph import TrainDeleteLBPH
from save_delete_encoding import SaveDeleteEncodings

class DeleteFace:
    
    def __init__(self, model):
        self.model = model
        
    def getModel(self):
        return self.model
    
    def deleteFace(self, name, uniqueId):
    
        success1, _ = SaveDeleteEncodings.delete_encodings(name, uniqueId) # method to remove encodings from saved encodings
        if success1 == False:
            return False
        success2 = TrainDeleteLBPH(self.getModel()).delete_face_lbph(name, uniqueId) # method to remove person from trained lbph model
        if success2 == False:
            return False
        else:
            return True