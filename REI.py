import cv2
from cvzone.PoseModule import PoseDetector

video = cv2.VideoCapture('vd3.mp4')

#webcam
# video = cv2.VideoCapture(0)

detector = PoseDetector()

base = 510

while True:
    success, img = video.read()
    img = cv2.resize(img,(720,1000))
    #webcam
    #img = cv2.resize(img, (1000, 720))
    w, h, _ = img.shape

    results = detector.findPose(img)

    lmlist, bboxInfo = detector.findPosition(img,draw=False)

    cv2.rectangle(img,(0,base),(h,base+20),(255,0,0),-1)

    if len(lmlist) >= 1:
        x,y,w,h = bboxInfo['bbox']

        pex = lmlist[31][0]
        pey = lmlist[31][1]

        mox = lmlist[20][0]
        moy = lmlist[20][1]

        distanciaPe = base-pey

        distanciaMo = base - moy

        # 274 px = 53 cm
        #distCm = (distancia*53)/274
        # 164 px = 47 cm
        #distCm = (distancia*47)/164

        #proporção
        # 100 px = 53 cm
        distCmPe = (distanciaPe*53)/100
        distCmMo = (distanciaMo * 53) / 100

        if distanciaPe <0:
            distanciaPe=0
        if distCmPe <0:
            distCmPe=0

        if distanciaMo <0:
            distanciaMo=0
        if distCmMo <0:
            distCmMo=0

        cv2.putText(img,f'Cm: {int(distCmPe)}',(pex+20,pey),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(img, f'Cm: {int(distCmMo)}', (mox+20, moy), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 2)
        # print(distancia)

    cv2.imshow('Imagem',img)
    cv2.waitKey(1)