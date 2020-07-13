import cv2

video = cv2.VideoCapture(0)
while True:
    flag,img = video.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # to detect face
    faceFile = cv2.CascadeClassifier('face.xml')
    eyeFile = cv2.CascadeClassifier('eye.xml')
    faces = faceFile.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),2)

        face_gray = gray[y:y+h,x:x+w]   #face area of gray img
        face_color = img[y:y + h, x:x + w]  #face area of colored img
        # to detect eye
        eyes = eyeFile.detectMultiScale(face_gray, 1.1, 5)
        for (x1, y1, w1, h1) in eyes:
            cv2.rectangle(face_color, (x1, y1), ((x1 + w1), (y1 + h1)), (0, 255, 255), 2)

    cv2.imshow('MY VIDEO', img)
    if cv2.waitKey(1)==27:
        break

video.release()
cv2.destroyAllWindows()