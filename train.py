# train.py
import cv2, sys, numpy, os
import sqlite3

size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'

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


try:
    fn_name = raw_input('Enter user name: ')
except:
    print("You must provide a name")
    sys.exit(0)

name = "\""+fn_name +"\""
insertOrUpdate(nextid(),name)
path = os.path.join(fn_dir, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (112, 92)
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)

pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     if n[0]!='.' ]+[0])[-1] + 1


count = 0
pause = 0
count_max = 20
while count < count_max:

    rval = False
    while(not rval):
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    height, width, channels = frame.shape

    frame = cv2.flip(frame, 1, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    faces = haar_cascade.detectMultiScale(mini)

    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]

        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
            1,(0, 255, 0))

        if(w * 6 < width or h * 6 < height):
            print("Face too small")
        else:

            if(pause == 0):

                print("Saving training sample "+str(count+1)+"/"+str(count_max))

                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                pin += 1
                count += 1

                pause = 1

    if(pause > 0):
        pause = (pause + 1) % 5
    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()