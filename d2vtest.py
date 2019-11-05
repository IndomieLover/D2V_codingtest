import cv2
import numpy as np
import math
from math import atan

class imageProcess:
    def __init__(self):
        self.imageNumber = 1 # to scroll through image
        self.meanvalue= None
        # variables for hough line
        self.rho = 1
        self.theta = np.pi / 180  
        self.threshold = 15
        self.min_line_length = 10
        self.max_line_gap = 3

        # variables to store frame
        self.resultFrame = None
        self.rawFrame = None
        self.arrowImg = None

        # width and height for image
        self.width = 0
        self.height = 0
        self.kernel = np.ones((5,5),np.uint8)
        #set output for text
        self.outputTxt=open("./result/output.txt", "w+")
        self.outputTxt.write("FILENAME\t\tDIRECTION\n")
        

    def main(self):
        while True:
            self.gradient=[]
            self.rawFrame = cv2.imread("./testimg/%i.JPG" %self.imageNumber)
            self.width, self.height, _= self.rawFrame.shape

            self.processImage() #process image by increasing contrast then blur it then search 
                                #   for the edges using canny and detect lines by hough line

            self.getGradient() # get gradient for every line that is detected

            self.drawArrow() # draw arrow to the 2 most similar gradient on different sides


            cv2.imwrite("./result/result_%i.JPG" %self.imageNumber , self.rawFrame) # write output image
            self.imageNumber= self.imageNumber+1
            if self.imageNumber==10:
                self.outputTxt.write("*direction is in degree in respect of x-axis")
                self.outputTxt.close()
                break

    def processImage(self):
        #increase contrast
        
        grayFrame = cv2.cvtColor(self.rawFrame, cv2.COLOR_BGR2GRAY)

        #crop image to the center
        croppedImg = grayFrame[self.width/2-40:self.width/2+40,self.height/2-40:self.height/2+40]
        
        img = cv2.GaussianBlur(croppedImg,(3,3), 4)
        template=np.zeros(img.shape[:],dtype=np.uint8)
        value= cv2.mean(grayFrame[self.width/2-10:self.width/2+10,self.height/2-10:self.height/2+10])
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
        template=cv2.morphologyEx(template, cv2.MORPH_CLOSE, self.kernel)
        template=cv2.dilate(template,self.kernel,iterations=3)
        template=cv2.bitwise_and(template,croppedImg,dst=None,mask=None)
        template= cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        lab= cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        template = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # grayFrame = cv2.GaussianBlur(template,(3,3), 1)

        #search for edges
        grayFrame = cv2.Canny(grayFrame,200,400, apertureSize = 3)
        self.resultFrame = cv2.dilate(grayFrame,(3,3),iterations=1)

        #search for lines
        self.arrowImg = cv2.HoughLinesP(self.resultFrame, self.rho, self.theta, self.threshold, np.array([]),self.min_line_length, self.max_line_gap)

    def getGradient(self):
        #get gradient for every lines detected
        if self.arrowImg is not None:
            iterate=1
            for arrows in self.arrowImg:
                x1, y1, x2, y2=arrows[0]
                self.gradient.append(0 if x2-x1==0 else float(y2-y1)/(x2-x1))
                iterate= iterate+1
                
    def drawArrow(self):
        #draw arrow from comparing gradients
        for i in range(0,len(self.gradient)):
            for j in range(i+1,len(self.gradient)):
                if(abs(float(self.gradient[i]-self.gradient[j]))<0.30):
                    x1, y1, x2, y2=self.arrowImg[i][0]
                    _x1, _y1, _x2, _y2=self.arrowImg[j][0]
                    length1=math.sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1))
                    length2=math.sqrt((_y2-_y1)*(_y2-_y1)+(_x2-_x1)*(_x2-_x1))
                    length3=math.sqrt((_y2-y1)*(_y2-y1)+(_x2-x1)*(_x2-x1)) 
                          
                    if ((abs(x1-_x1)>15 and abs(x2-_x2)>15) or (abs(y1-_y1)>15 and abs(y2-_y2)>15)) and abs(length1-length2)<10 and length3 > 20 and length1>10 and length2 >10:
                        direction=atan(0 if (_x2-x1)==0 else float(y1-_y2)/(_x2-x1))*180/np.pi
                        
                        x1=int(x1+(x1-_x2)/length3*50)
                        y1=int(y1+(y1-_y2)/length3*50)
                        _x2=int(_x2+(_x2-x1)/length3*30)
                        _y2=int(_y2+(_y2-y1)/length3*30)
                        
                        self.outputTxt.write("{}.JPG\t\t\t{}\n".format(self.imageNumber,direction))
                        if _x2>x1:
                            cv2.arrowedLine(self.rawFrame,(self.width/2-40+x1,self.height/2-40+y1),(self.width/2-40+_x2,self.height/2-40+_y2),(0,0,255),2, tipLength=0.08)
                        else:
                            cv2.arrowedLine(self.rawFrame,(self.width/2-40+_x2,self.height/2-40+_y2),(self.width/2-40+x1,self.height/2-40+y1),(0,0,255),2, tipLength=0.08)
                        return

if __name__=='__main__':
    imageProcess().main()
        

