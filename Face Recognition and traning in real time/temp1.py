# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 22:49:05 2017

@author: DELL
"""
import os
import cv2
path1 = 'D:/Study/Sem 5/Mini Project Docs/faces94/female'
files = os.listdir(path1)
i = 3
for file in files:
    path = path1 +'/'+str(i)
    files1=os.listdir(path)
    k=1
    for file in files1:
        image = cv2.imread(path+'/person.'+str(40+i)+'.'+str(k)+'.jpg')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('person.'+str(i+81)+'.'+str(k)+'.jpg',gray_image)
        cv2.waitKey(0)                 # Waits forever for user to press any key
        cv2.destroyAllWindows()      
        k=k+1
    i = i+1
  