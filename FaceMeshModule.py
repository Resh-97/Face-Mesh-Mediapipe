import cv2
import mediapipe as mp
import time
#cam = cv2.VideoCapture('Videos/3.mp4')
cam = cv2.VideoCapture(0)
class faceMesh():

    def __init__(self,mode = False, maxFace = 2, detectionConfidence = 0.5, trackConfidence = 0.5 ):
        self.mode = mode
        self.maxFace = maxFace
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode,self.maxFace,self.detectionConfidence,self.trackConfidence )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1,circle_radius =1)

    def findFaceMesh(self,img, draw =True):
        imgRGB  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Original Images are BGR
        self.results = self.faceMesh.process(imgRGB)  #Processes RGB and returns two fields: "multi_face_landmarks"
        faces = []
        if self.results.multi_face_landmarks:
            for faceLm in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLm, self.mpFaceMesh.FACE_CONNECTIONS,self.drawSpecs,self.drawSpecs) #Draws the keypts & connections
                face = []
                for id,lm in enumerate(faceLm.landmark):
                    #print(lm)
                    height, width, channel = img.shape
                    x_center, y_center = int(lm.x * width), int(lm.y * height)
                    #cv2.putText(img, str(id),(x_center,y_center), cv2.FONT_HERSHEY_PLAIN, 0.5,(255,0,255),1)
                    #print(id, x_center, y_center)
                    face.append([id, x_center, y_center])
                faces.append(face)
        return img,faces

def main():
    prevTime = 0
    currTime = 0
    detector = faceMesh()
    while True:
        success, img = cam.read()
        #img = cv2.resize(img,(640,480))
        img, faces= detector.findFaceMesh(img,draw =True)
        if len(faces)!=0:
            print(faces[0])
        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__== '__main__':
    main()
