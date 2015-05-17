
import cv
import numpy as np
import cv2
import math
from Queue import Queue
from os import listdir, makedirs
from os.path import isfile, join, exists

class RegionGrower:

    myImage = None
    myThresholdedImage = None

    myFilledMatrix = None
    myVisitedMatrix = None
    myHeight = 0
    myWidth = 0

    my8ConnectionNeeded = True

    myPoints2BVisited = None

    myMaxVal = 0
    myMinVal = 0

    def __init__(self, anImageMat_in):

        aNumChannels = anImageMat_in.shape[2]
        if aNumChannels == 4:
            self.myImage = cv2.cvtColor(anImageMat_in, cv2.COLOR_BGRA2GRAY)
        elif aNumChannels == 3:
            self.myImage = cv2.cvtColor(anImageMat_in, cv2.COLOR_BGR2GRAY)
        elif aNumChannels == 1:
            pass
        else:
            raise Exception("Unexpected file type for file %s. Type is %s" % (anImageMat_in, anImageMat_in[:,:].type))

        self.myThresholdedImage = cv2.adaptiveThreshold(self.myImage,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

        self.myHeight, self.myWidth = self.myImage.shape[:2]
        self.myFilledMatrix = np.asarray(cv.CreateMat(self.myHeight, self.myWidth, cv.CV_8UC1))
        self.myVisitedMatrix = np.asarray(cv.CreateMat(self.myHeight, self.myWidth, cv.CV_8UC1))

        self.myPoints2BVisited = Queue(maxsize=0)

        self.myMaxVal = max(np.max(self.myImage[0, :]), np.max(self.myImage[:, 0]))
        self.myMinVal = min(np.min(self.myImage[0, :]), np.max(self.myImage[:, 0]))
        print self.myMaxVal
        print self.myMinVal
        

        
    def Grow(self, x, y):

        self.myPoints2BVisited.put((x, y))

        while not self.myPoints2BVisited.empty():
            aPoint = self.myPoints2BVisited.get_nowait()
            self.visitPixel(aPoint[0], aPoint[1])

    def GetVisitedMatrix(self):
        return self.myFilledMatrix
        
    def visitPixel(self, x, y):
        
        self.myFilledMatrix.itemset((x, y), 255)
        
        for i in range(x - 1, x + 2): 
            for j in range(y - 1, y + 2):
                if i >= 0 and j >= 0 and i < self.myHeight and j < self.myWidth:
        
                    if self.myVisitedMatrix.item(i, j) == 1:
                        continue

                    self.myVisitedMatrix.itemset((i, j), 1)

                    if i == x and j == y:
                        continue

                    if not self.my8ConnectionNeeded:
                        aDist = exp(i - x, 2) + exp(j - y, 2)
                        if aDist > 1:
                            continue

                    if self.myImage.item(i, j) <= self.myMaxVal and self.myImage.item(i, j) >= self.myMinVal:
                        self.myPoints2BVisited.put((i, j))
    

class DressExtractor:

    myImage = None
    myDressOnlyMat = None

    myTransparencyChannel = None
    
    myImagePath = ""

    def __init__(self, anImageName_in):

        self.myImagePath = anImageName_in
        self.myImage = cv2.imread(anImageName_in)

    def MakeDressImage(self):

        self.Extract()
        self.WriteImage()

    def Extract(self):

        rg = RegionGrower(self.myImage)
        rg.Grow(0, 0)
        myTransparencyChannel = rg.GetVisitedMatrix()
        
        height, width = self.myImage.shape[:2]
        #self.myDressOnlyMat = self.myImage[:,:].copy()

        aDressBWImg = cv2.bitwise_not(myTransparencyChannel)
        aDressOnlyMatBGR = cv2.bitwise_and(self.myImage, self.myImage, mask = aDressBWImg)

        b,g,r = cv2.split(aDressOnlyMatBGR)
        a = aDressBWImg

        self.myDressOnlyMat = cv.CreateMat(height, width, cv.CV_8UC4)
        self.myDressOnlyMat = cv2.merge([b,g,r,a])

    def WriteImage(self):

        aLastSlash = self.myImagePath.rfind("/")
        aNewFileName = self.myImagePath[0 : aLastSlash] + "/Extracted" + self.myImagePath[aLastSlash : self.myImagePath.rfind(".")] + ".png"
        params = list()
        params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
        params.append(9)
        cv2.imwrite(aNewFileName, self.myDressOnlyMat, params)


class BatchDressExtractor:

    myFileList = None
    myOutputFolder = ""

    def __init__(self, aFolder_in):

        self.myFileList = [ aFolder_in + "/" + f for f in listdir(aFolder_in) if isfile(join(aFolder_in,f)) ]
        print "Number of files in folder %s is : %s" % (aFolder_in, len(self.myFileList))
        self.myOutputFolder = aFolder_in + "/Extracted"
        print "Output folder is : %s" % (self.myOutputFolder)

        if not exists(self.myOutputFolder):
            print "The folder " + self.myOutputFolder + " not existing. Creating new"
            makedirs(self.myOutputFolder)

    def Process(self):
        [self.ExtractFile(f) for f in self.myFileList]

    def ExtractFile(self, aFileName_in):

        try:
            print "Processing file :" + aFileName_in
            de = DressExtractor(aFileName_in)
            de.MakeDressImage()
        except:
            e = sys.exc_info()[0]
            print "Exception caught : %s" % e
     


#img = cv2.imread("/home/unn/Downloads/Dress/churidar.jpg")

#rg = RegionGrower(img)
#rg.Grow(0, 0)

#de = DressExtractor("/home/unn/Downloads/Dress/churidar.jpg")
#de.MakeDressImage()

bde = BatchDressExtractor("/home/unn/Downloads/Dress")
bde.Process()


#cv2.imshow('dress', de.GetDressOnlyMat())

'''params = list()
params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
params.append(9)


cv2.imwrite("/home/unn/Downloads/Dress/churidar_extracted.png", de.GetDressOnlyMat(), params)'''

cv2.waitKey(0)

'''
im_array_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

th2 = cv2.adaptiveThreshold(im_array_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

cv2.imshow('dress',th2)

cv2.waitKey(0)
'''


'''
cap = cv2.VideoCapture('/home/unn/Desktop/videoplayback.avi')
     
fgbg = cv2.BackgroundSubtractorMOG()

img = cv2.imread("/home/unn/Downloads/Dress/churidar3.jpg")

img_bg = cv2.imread("/home/unn/Desktop/white_bg.png")

while(1):
    ret, frame = cap.read()

    
    fgmask = fgbg.apply(img, img_bg)

    print fgmask

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

'''

'''
img = cv.LoadImage("/home/unn/Downloads/Dress/churidar3.jpg")
cv.NamedWindow("opencv")
cv.ShowImage("opencv",img)

fgbg = cv2.BackgroundSubtractorMOG()

img_np = np.asarray(img[:,:])
im_array_gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)

im_array = np.asarray( im_array_gray )

fgmask = fgbg.apply(im_array)

print fgmask

img2 = np.zeros_like(img)

height, width = fgmask.shape[:2]
fgmask_cv = cv.CreateMat(height, width, cv.CV_8UC1)

print img[:,:].type

print("width = %s height = %s" % (width, height))

cv.SetData(fgmask_cv, fgmask.data)

cv.NamedWindow("opencv_bg")
cv.ShowImage("opencv_bg",fgmask_cv)

cv.WaitKey(0)

'''

