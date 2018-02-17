import cv2, numpy, os, shutil
import pyttsx
import sqlite3
size = 2
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'
speech_engine = pyttsx.init('sapi5') 
speech_engine.setProperty('rate', 150)
p='temp'
model=""
def getdetail(Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="Select block from people where Name="+str(Name)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1;
    if(isRecordExist==1):
        if(row[0] == "yes"):
            return True
    else:
        return False
    conn.execute(cmd)
    conn.commit()
    conn.close()
    
def speak(text):
	speech_engine.say(text)
	speech_engine.runAndWait()

def insertOrUpdateB(Id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="Select Name,Age,Gender from people where Name="+str(Name)
    cursor=conn.execute(cmd)
    isRecordExist=0
    c = "yes"
    c = "\""+c +"\""
    for row in cursor:
        isRecordExist=1;
    if(isRecordExist==1):
        cmd="update people set Name="+str(Name)+" where id="+str(Id)
    else:
        cmd="insert into people(Id,Name,block) values("+str(Id)+", "+str(Name)+", "+c+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
    
def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="Select Name,Age,Gender from people where Name="+str(Name)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1;
    if(isRecordExist==1):
        cmd="update people set Name="+str(Name)+" where id="+str(Id)
    else:
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
        ids=row
    conn.close()
    max=0
    for i in ids:
        if(i>max):
            max=i
    return max+1

print('Training...')


(images, lables, names, id) = ([], [], {}, 0)

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
            lable = id

            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

(images, lables) = [numpy.array(lis) for lis in [images, lables]]

model = cv2.createLBPHFaceRecognizer()
model.train(images, lables)



j=0
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)
flag=False
flag1=False
flag2=False
flag3=False
person1 = ""
count = 0
pause = 0
count_max = 20
#pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     #if n[0]!='.' ]+[0])[-1] + 1
if not os.path.isdir(p):
    os.mkdir(p)
while True:
    
    rval = False
    while(not rval):
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    height, width, channels = frame.shape
    frame=cv2.flip(frame,1,0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
    
    faces = haar_cascade.detectMultiScale(mini)
    if(flag3==False):
        faces = sorted(faces, key=lambda x: x[3])
        flag3==True
    
    for i in range(len(faces)):
        face_i = faces[i]

        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if(count < count_max):
            if(w * 6 < width or h * 6 < height):
                print("Face too small")
            else:   
                if(pause == 0):
        
                    print("Saving training sample "+str(count+1)+"/"+str(count_max))
                    #cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
                    cv2.imwrite(p+"/"+str(count+1)+".png", face_resize)
                    #pin += 1
                    count += 1
        
                    pause = 1
        
            if(pause > 0):
                pause = (pause + 1) % 5
            
        else:    
            if prediction[1]<60:
                cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                if(flag2==False):
                    speak("Hello "+names[prediction[0]])
                    flag2=True
                if(flag1==False):
                    data = "\""+names[prediction[0]] +"\""
                    if(getdetail(data)):
                        speak("This person is blocked")
                        flag1=True
                j=0
                 
            elif(j>50):
                flag=True
                flag1=True
                flag2=True
                speak("New person detected")
                speak("Would you like to open door and store person in database")
                choice = raw_input("Enter choice Y/N: ")
                
                
                if(choice == "Y"):
                    speak("Please insert person details")
                    name = raw_input('Enter name: ')
                    if not os.path.isdir(p):
                        os.mkdir(p)
                    """folder = os.listdir(p)
                    for fn in folder:
                        print fn"""
                    #os.rename(p, name)
                    path = os.path.join(fn_dir, name)
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    files = os.listdir(p)
                    files.sort()
                    
                    for f in files:
                        src = p+"/"+f
                        dst = fn_dir+"/"+name+"/"+f
                        shutil.move(src,dst)
                    """src = name
                    dst = fn_dir+"/"
                    shutil.move(src,dst)"""
                    name1 = "\""+name +"\""
                    insertOrUpdate(nextid(),name1)
                    #webcam.release()
                    #cv2.destroyAllWindows()  
                else:
                    """webcam.release()
                    cv2.destroyAllWindows()"""  
                    speak("Person is to be added in block list")
                    speak("Please insert person details")
                    name = raw_input('Enter name: ')
                    if not os.path.isdir(p):
                        os.mkdir(p)
                    """folder = os.listdir(p)
                    for fn in folder:
                        print fn"""
                    #os.rename(p, name)
                    path = os.path.join(fn_dir, name)
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    files = os.listdir(p)
                    files.sort()
                    
                    for f in files:
                        src = p+"/"+f
                        dst = fn_dir+"/"+name+"/"+f
                        shutil.move(src,dst)
                    
                    name1 = "\""+name +"\""
                    insertOrUpdateB(nextid(),name1)
                j=0
                
        if(flag==True):
            break
        j+=1
    
    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if key == 27:
        webcam.release()
        cv2.destroyAllWindows()
        break