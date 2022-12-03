import cv2

#This loads pre-trained data on fontal faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Original Image
#img=cv2.imread('weeknd.jpg')

#Video
webcam=cv2.VideoCapture("gif.gif")
    

#iterate over each frame of the video
while True:
    read_frame, frame = webcam.read()
#greyscaled img
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Detect Faces
#takes the classifier and detects the image regardless of scale and return obj as a list of rectrangles (xy for top left corner of object, zh for bottem right point of object)
    face_cordinates = trained_face_data.detectMultiScale(greyscaled_img)

#loops through the list of cordinates and draws rectangles on the original img
    for (x,y,z,h) in face_cordinates:
        cv2.rectangle(frame, (x, y), (x+z,y+h), (0,255,0), 2)
#print(face_cordinates)



    cv2.imshow('Programe face detector',frame)
    key = cv2.waitKey(100)

    #if click Q, quit
    if key == 81 or key ==113:
        break

webcam.release()
print("code completed")