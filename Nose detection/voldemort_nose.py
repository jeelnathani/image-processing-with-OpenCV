import cv2

image = cv2.imread('voldemort_harry.jpg',1)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#noseFile = cv2.CascadeClassifier('./cascade_files/haarcascade_mcs_nose.xml')
noseFile = cv2.CascadeClassifier('nose.xml')
noses = noseFile.detectMultiScale(gray,1.5,5)

for (y,x,w,h) in noses:
    cv2.rectangle(image,(y,x),((y+h),(x+w)),(0,0,255),2)

cv2.imshow('colored',image)
#cv2.imshow('B-and-W',gray)
cv2.waitKey(0)
