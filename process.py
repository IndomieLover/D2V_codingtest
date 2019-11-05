import cv2
import numpy as np
import math

def nothing(x):
    pass
def drawArrow(gradient,lines,croppedImg):
    for i in range(0,len(gradient)):
        for j in range(i+1,len(gradient)):
            if(abs(float(gradient[i]-gradient[j]))<0.30):
                
                x1, y1, x2, y2=lines[i][0]
                _x1, _y1, _x2, _y2=lines[j][0]
                length1=math.sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1))
                length2=math.sqrt((_y2-_y1)*(_y2-_y1)+(_x2-_x1)*(_x2-_x1))
                length3  =math.sqrt((_y2-y1)*(_y2-y1)+(_x2-x1)*(_x2-x1))     
                tempGrad = float(_y2-y1)/(_x2-x1)   
                print(gradient[i],gradient[j], tempGrad)  
                if ((abs(x1-_x1)>25 and abs(x2-_x2)>25) or (abs(y1-_y1)>25 and abs(y2-_y2)>25)) and abs(length1-length2)<5  and abs(gradient[i]-tempGrad)>1:
                    print(length1,length2)
                    # x1=int(x1+(x1-_x2)/math.sqrt(length3**2)*30)
                    # y1=int(y1+(y1-_y2)/math.sqrt(length3**2)*30)
                    cv2.arrowedLine(croppedImg,(x1,y1),(_x2,_y2),(0,255,0),2)
                    print(gradient[i]-gradient[j])
                    return
num=1

# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("CannyUp", "Trackbars", 156, 500, nothing)
# cv2.createTrackbar("CannyDown", "Trackbars", 708, 1000, nothing)

while True:
    # CannyUp = cv2.getTrackbarPos("CannyUp", "Trackbars")
    # CannyDown = cv2.getTrackbarPos("CannyDown", "Trackbars")
    gradient=[]
    rawFrame = cv2.imread("./testimg/%i.JPG" %num)
    rawFrame = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2GRAY)

    width, height= rawFrame.shape
    kernel = np.ones((5,5),np.uint8)
    croppedImg = rawFrame[width/2-40:width/2+40,height/2-40:height/2+40]
    img = cv2.GaussianBlur(croppedImg,(3,3), 4)
    template=np.zeros(img.shape[:],dtype=np.uint8)
    value= cv2.mean(rawFrame[width/2-10:width/2+10,height/2-10:height/2+10])
    _, img = cv2.threshold(img, float(value[0])-30, float(value[0])+50, cv2.THRESH_BINARY)

    xcenter=30
    ycenter=30
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
    template=cv2.bitwise_and(template,croppedImg,dst=None,mask=None)
    
    rawFrame= cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    lab= cv2.cvtColor(rawFrame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    template = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    grayFrame = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # grayFrame = cv2.medianBlur(grayFrame,5)
    grayFrame = cv2.bilateralFilter(grayFrame,21,100,100)
    # grayFrame = cv2.GaussianBlur(grayFrame,(19,19), 1)

    grayFrame = cv2.Canny(grayFrame,50,150, apertureSize = 3)
    # grayFrame = cv2.dilate(grayFrame,(3,3),iterations=1)
    # grayFrame = cv2.morphologyEx(grayFrame,cv2.MORPH_CROSS,kernel)


    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(grayFrame, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    print(lines)
    if lines is not None:
        iterate=1
        for line in lines:
            x1, y1, x2, y2=line[0]
            gradient.append(0 if x2-x1==0 else float(y2-y1)/(x2-x1))
            # print(gradient[iterate-1])
            cv2.line(croppedImg,(x1,y1),(x2,y2),(255,0,0),2)
            iterate= iterate+1
    print(gradient)
    drawArrow(gradient,lines,croppedImg)
    # for i in range(0,len(gradient)):
    #     for j in range(i+1,len(gradient)):
    #         if(abs(float(gradient[i]-gradient[j]))<0.2):
    #             x1, y1, x2, y2=lines[i][0]
    #             _x1, _y1, _x2, _y2=lines[j][0]
    #             cv2.line(croppedImg,(x1,y1),(_x2,_y2),(0,255,0),2)
    #             print(gradient[i]-gradient[j])
    #             print(i)
    #             print(j)
    #             break

    # print(gradient)
    # lines = cv2.HoughLines(grayFrame,1,np.pi/180,1)
    # if lines is not None:
    #     for rho,theta in lines[0]:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a*rho
    #         y0 = b*rho
    #         x1 = int(x0 + 1000*(-b))
    #         y1 = int(y0 + 1000*(a))
    #         x2 = int(x0 - 1000*(-b))
    #         y2 = int(y0 - 1000*(a))

    #         cv2.line(croppedImg,(x1,y1),(x2,y2),(0,0,255),2)
    # grayFrame= cv2.morphologyEx(grayFrame, cv2.MORPH_CLOSE, kernel,2)
    cv2.imshow("img",croppedImg)
    cv2.imshow("gray",grayFrame)
    cv2.imshow("template",template)
    cv2.moveWindow("gray", 100, 0)
    cv2.moveWindow("img", 200, 0)
    
    
    if cv2.waitKey(0) or cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyWindow("img")
        cv2.destroyWindow("gray")
        num = num + 1   
        if num==10:
            break