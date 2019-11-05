import cv2
import numpy as np
font = cv2.FONT_HERSHEY_COMPLEX
imgraw = cv2.imread("./testimg/9.JPG", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.uint8)
width, height=imgraw.shape
imgcrop = imgraw[width/2-50:width/2+50,height/2-50:height/2+50]
img = cv2.GaussianBlur(imgcrop,(3,3), 4)
template=np.zeros(img.shape[:],dtype=np.uint8)
value= cv2.mean(imgraw[width/2-10:width/2+10,height/2-10:height/2+10])
_, img = cv2.threshold(img, float(value[0])-30, float(value[0])+50, cv2.THRESH_BINARY)

xcenter=30
ycenter=30
imgresult=cv2.bitwise_and(img,imgcrop)
w, h=img.shape
best=None
_, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    M = cv2.moments(cnt)
    cX=0
    cY=0
    if M["m00"] != 0:
        cX= int(M["m10"] / M["m00"])
        cY= int(M["m01"] / M["m00"])
    if abs(w/2-cX)<=xcenter and abs(h/2-cY)<=ycenter:
        best=cnt
        xcenter=abs(w/2-cX)
        ycenter=abs(h/2-cY)

    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

cv2.drawContours(template, [best], 0, (255), -1)
template=cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel)
template=cv2.dilate(template,kernel,iterations=3)
template=cv2.bitwise_and(template,imgcrop,dst=None,mask=None)
# template=cv2.Canny(template,200,600)

cv2.imshow("shapes", template)
cv2.imshow("result", imgresult)
cv2.waitKey(0)
cv2.destroyAllWindows()