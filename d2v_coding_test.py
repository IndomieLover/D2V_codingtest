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


            cv2.imwrite("./result/result_{}.JPG".format(self.imageNumber) , self.rawFrame) # write output image
            self.imageNumber= self.imageNumber+1
            if self.imageNumber==10: #image from 1-9, 10 is out of bound
                self.outputTxt.write("#direction is in degree in respect of x-axis")
                self.outputTxt.close()
                break

    def processImage(self):
        #increase contrast
        labImage = cv2.cvtColor(self.rawFrame, cv2.COLOR_BGR2LAB)
        l,a,b= cv2.split(labImage)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrastL = clahe.apply(l)
        labImage = cv2.merge((contrastL,a,b))
        contrastFrame = cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR)
        
        grayFrame = cv2.cvtColor(contrastFrame, cv2.COLOR_BGR2GRAY)

        #crop image to the center
        croppedImg = grayFrame[self.width/2-40:self.width/2+40,self.height/2-40:self.height/2+40]
        grayFrame = cv2.GaussianBlur(croppedImg,(3,3), 1)

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
                    tempGrad=0
                    if _x2-x1!=0:
                        tempGrad = float(_y2-y1)/(_x2-x1) 

                    if ((abs(x1-_x1)>15 and abs(x2-_x2)>15) or (abs(y1-_y1)>15 and abs(y2-_y2)>15)) and abs(length1-length2)<10 and length3 > 20 and length1>10 and length2 >10 and abs(self.gradient[i]-tempGrad)>1:
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
        

