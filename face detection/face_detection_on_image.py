import cv2

image = cv2.imread('face_hero.jpeg',1)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faceFile = cv2.CascadeClassifier('face.xml')  #linking xml file to python file
# always pass gray scale img only
faces = faceFile.detectMultiScale(gray,1.3,5) #(var,scale factor = 1.1-1.9, minimum neighbours = 1-9)
print(faces)  #returns (x coordinate,y coordinate,width,height)
for (y,x,w,h) in faces:
    cv2.rectangle(image,(y,x),((y+h),(x+w)),(0,255,255),2) #(var, 1st pt,2nd pt,color,width)

cv2.imshow('colored',image)
cv2.imshow('B-and-W',gray)
cv2.waitKey(0)