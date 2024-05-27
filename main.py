import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()


listImg = os.listdir("Images")
print(listImg)
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgBg = cv2.resize(img, (640, 480))
    imgList.append(imgBg)
    
print(len(imgList))

indexImg = 0

pTime =0
while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg],0.8)
    
    imgStacked = cvzone.stackImages([img, imgOut], 2,1)
    
    cTime = time.time()      
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(imgStacked, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) 
        
    cv2.imshow("ImageOut", imgStacked)
    key = cv2.waitKey(1) 
    if key == ord('a'):
        if indexImg>0:
            indexImg -= 1
    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg +=1
    elif key == ord('q'):
        break    