import cv2
import mediapipe as mp
import time
cam = cv2.VideoCapture('Videos/2.mp4')
#cam = cv2.VideoCapture(0)
prevTime = 0
currTime = 0

mpFaceMesh = mp.solutions.face_mesh
#Parameters: mode, max no of faces, detection & tracking confidence
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawSpecs = mpDraw.DrawingSpec(thickness=1,circle_radius =1)


while True:
    success, img = cam.read()
    #img = cv2.resize(img,(480,640))
    img = cv2.resize(img,(640,480))
    imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    print(results.multi_face_landmarks)

    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLm, mpFaceMesh.FACE_CONNECTIONS,drawSpecs,drawSpecs) #Draws the keypoints
            for id,lm in enumerate(faceLm.landmark):
                #print(lm)
                height, width, channel = img.shape
                x_center, y_center = int(lm.x * width), int(lm.y * height)
                print(id, x_center, y_center)


    currTime =time.time()
    fps = 1/ (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
