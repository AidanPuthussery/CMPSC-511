import cv2 
import numpy as np
import HandTrackingMod as htm
import time
import autopy

##################

wCam, hCam = 640, 480
frameR = 100
##################



cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0


detector = htm.handDector(maxHands=1)

wScr, hScr = autopy.screen.size()

while True:
    #1. FInd hand landmarks 
    sucess, img = cap.read()

    if not sucess:
        print("Failed to capture image")
        continue  # Skip the current iteration of the loop

    img = detector.findHands(img)
    lmList, bbox = detector.findPostion(img)

    #2. get the tip of the index and middle finger
    
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        
        #print(x1, y1, x2, y2)

        #3. Check whchi fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
    
        #4. only Idex finger : Moving mode 
        if fingers[1] == 1 and fingers[2] == 0:

            #5. Convert cooridnates
            cv2.rectangle(img, (frameR, frameR),(wCam-frameR, hCam- frameR), (255,0,255), 2)
            x3 = np.interp(x1, (frameR, wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0,hScr)) 
 
            #6. Smoothen Values 
            #7. Move Mouse
            autopy.mouse.move(wScr-x3, y3)
            cv2.circle(img, (x1,  y1), 15, (255,0 ,255), cv2.FILLED)
    #8. Both Index and Middle fingers are up: Clicking mode 
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, _ = detector.findDistance(8, 12, img)
            print(length)
            if length < 30:
                autopy.mouse.click()

    #9. Find the distances between each finger 
    #10. Click mouse if distnace short 
        

    #11. Frame rate 
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_COMPLEX,3 , (255,0,0), 3)
    #12. Display 
    cv2.imshow("Image", img)
    cv2.waitKey(1)
