from face_recognition import *
from os.path import *
from os import makedirs,listdir,remove
from pickle import *
from cvlib.object_detection import YOLO
from keyboard import is_pressed
import cv2


known_face_encodings = []

if isfile("data\\data.pkl"):
    with open("data\\data.pkl","rb") as file:
        known_face_encodings.extend(load(file))
        file.close()



images = listdir("faces")
length = len(images)

if length>0:
    print(length,"images found..")

    for filename in images:
        print("Encoding",filename)
        image = cv2.imread(f"faces\\{filename}")
        locations = face_locations(image)

        if len(locations) == 0:
            print("Face Not Found...")
            continue

        if len(locations)>1:
            print("More than 1 face found..")
            continue

        for location in locations:
            a,b,c,d = location
            image = cv2.rectangle(image,(d,a),(b,c),(51,225,225),2)
        
        cv2.imshow("Frame",image)
        cv2.waitKey(0)
        
        encodings = face_encodings(image)

        for index,encoding in enumerate(encodings):
            name = input(f"Name of {images[index]} image : ")
            known_face_encodings.append((name,encoding))
        
        cv2.imwrite(f"encoded\\{filename}",image)
        remove(f"faces\\{filename}")
        print(f"faces\\{filename} deleted..")

    with open("data\\data.pkl","wb") as file:
        dump(known_face_encodings,file)
        file.close()



precoded = [known[1] for known in known_face_encodings]
names = [known[0] for known in known_face_encodings]

camera = cv2.VideoCapture(0)


while not is_pressed("esc"):
    _,image = camera.read()
    encodings = face_encodings(image)
    locations = face_locations(image)

    if len(encodings)==0:
        print("No face found..")
        cv2.imshow("Frame",image)
        cv2.waitKey(1)
        continue

    for location in locations:
        a,b,c,d = location
        image = cv2.rectangle(image,(d,a),(b,c),(51,225,225),2)

    for encoding in encodings:
        distances = list(face_distance(precoded,encoding))
        min_ = min(distances)
        if min_>=0.6:
            cv2.imshow("Frame",image)
            cv2.waitKey(1)
            continue

        index = distances.index(min_)
        name = names[index]
        print(name,(1-min_)*100)
        cv2.imshow("Frame",image)
        cv2.waitKey(1)

    cv2.imshow("Frame",image)
    cv2.waitKey(1)


camera.release()
cv2.destroyAllWindows()