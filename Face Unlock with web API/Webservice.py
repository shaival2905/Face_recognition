# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:13:07 2018

@author: shaival
"""

from face_recognize import Recognize
from flask import Flask, request, render_template
from flask import jsonify
import os
import pickle
import cv2, dlib
from collections import defaultdict
from face_samples import FaceSample
from delete_face import DeleteFace
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
    
@app.route('/')
def home():
    return render_template('startApplication.html')
	
@app.route('/takeSample', methods = ['POST'])
def get_input():
    data={}
    all_face_encodings = defaultdict(dict) 
    count_sample_failure=0
    
    if request.method == 'POST':
        #takes unique id entered on the browser
        uniqueId = request.form['uid']
        #takes name entered on the browser
        name = request.form['name']
        data['id']=uniqueId
        data['name'] = name
        path = os.path.join(uniqueId,'dataset_faces.dat')
        if (os.path.isfile(path)):
            with open(path, 'rb') as f:
                all_face_encodings = pickle.load(f)
        
                if name in all_face_encodings.keys():
                    message= 'Name already exists!Please enter a new name!'
                    success='False'
                    return jsonify(message=message,success=success,data=data)
                
        #calls the function to take samples of the faces 
        output=FaceSample(detector,model).take_sample(name, uniqueId)
        
        while output == False and count_sample_failure < 2:
            count_sample_failure += 1
            output=FaceSample.take_sample(name, uniqueId)
            
       
        
        if output == False:
            message= 'Please upload video with face pointing towards camera'
            success='False'
    
        else:
            message ='Face samples taken succesfully'
            success='True'
            
        return jsonify(message=message,success=success,data=data)
    
@app.route('/faceRecognise', methods = ['POST'])
def recognise():
    data={}
    if request.method == 'POST':
        uniqueId = request.form['uid2']
          
        # call functions to unlock face
        output=Recognize(detector,model).recognize_face(uniqueId)
        data['status'] = output[0]
        data['name'] = output[1]
        data['id']=uniqueId
        
        if(output[0]=='false'):
            success='False'
            message='This Id is not registered'
        
        elif output[0] == 'notProper':
            success = 'False'
            message= 'Video not proper please upload another video'
            return jsonify(message=message,success=success,data=data)
        
        else:
            success='True'
            message='Successful'
        return jsonify(message=message,success=success,data=data)
  
    

@app.route('/videotrain', methods = ['POST'])
def trainvideo():
    data={}
    all_face_encodings = defaultdict(dict)
    
    if request.method == 'POST':
        uniqueId = request.form['uid2']
        name = request.form['name']
        
        if not os.path.isdir(uniqueId):
            os.mkdir(uniqueId)
        data['id']=uniqueId
        data['name'] = name    
        path = os.path.join(uniqueId,'dataset_faces.dat')
        if (os.path.isfile(path)):
            with open(path, 'rb') as f:
                all_face_encodings = pickle.load(f)
        
                if name in all_face_encodings.keys():
                    message= 'Name already exists!Please enter a new name!'
                    success='False'
                    return jsonify(message=message,success=success,data=data)
        
        f = request.files['file']
        _, f_extension = os.path.splitext(f.filename)
        
        # check for correct file type uploaded
        if(f_extension.lower() not in
                    ['.mp4' , '.avi', '.mkv', '.mov']):
            message= 'Invalid file type'
            success='False'
            return jsonify(message=message,success=success,data=data)        
            
        path = os.path.join(uniqueId,'train'+f_extension)
        f.save(path)  
        
        
        #calls the function to take samples of the faces 
        output=FaceSample(detector, model).take_sample(name, uniqueId, video=path)
        
        
        if output == False:
            message= 'Please upload video with face pointing towards camera'
            success='False'
        else:
            message= 'Face samples taken succesfully'
            success='True'
        return jsonify(message=message,success=success,data=data)

@app.route('/video', methods = ['POST'])
def recognisevideo():
    data={}
    if request.method == 'POST':
        uniqueId = request.form['uid2']
        
        if not os.path.isdir(uniqueId):
            return jsonify(status='unknown', person_name = 'null', id=uniqueId)
        
        f = request.files['file']
        _, f_extension = os.path.splitext(f.filename)
        
        # check for correct file type uploaded
        if(f_extension.lower() not in
                    ['.mp4' , '.avi', '.mkv', '.mov']):
            message= 'Invalid file type'
            success='False'
            return jsonify(message=message,success=success,data=data) 
        
        path = os.path.join(uniqueId,'recognise'+f_extension)
        f.save(path) 
        
        # call functions to unlock face
        output=Recognize(detector,model).recognize_face(uniqueId,video = path)

        # return if wrong file type uploaded
        data['status'] = output[0]
        data['name'] = output[1]
        data['id']=uniqueId
        
        if output[0]=='false':
            success='False'
            message='This Id is not registered'
        
        elif output[0] == 'notProper':
            success = 'False'
            message= 'Video not proper please upload another video'
            return jsonify(message=message,success=success,data=data)
       
        else:
            success='True'
            message='Successful'
        return jsonify(message=message,success=success,data=data)


@app.route('/deleteFace', methods = ['POST'])
def deleteface():
    
    if request.method == 'POST':
        data={}
        uniqueId = request.form['uid2']
        name = request.form['name']
    
        output = DeleteFace(model).deleteFace(name, uniqueId)
        data['id']=uniqueId
        data['name'] = name
        if output:
           message='Face deleted successfully'
           success='True'
        else:
            message= 'Face not found'
            success='False'
    return jsonify(message=message,success=success,data=data)

if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector() # get the dlib face detector
    model = cv2.face.LBPHFaceRecognizer_create(neighbors=10) # load LBPH face recogniser model
    app.run(debug = True, host = '0.0.0.0', threaded=True)
