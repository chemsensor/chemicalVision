import cv2
import numpy as np
import time
import math
try:
    import Tkinter as tk
    from tkFileDialog import askopenfilename
    from tkFileDialog import asksaveasfilename
except ImportError:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    from tkinter.filedialog import asksaveasfilename
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.spatial.distance import cdist 

lower_lim = np.array([0,0,0])
upper_lim = np.array([180,255,255])
MeanWindow=1
FrameByFrameToggle=False
ColorRectOver = False
ColorRectangle = False
ColorRect = (0,0,1,1)      
GreyRectOver = False
GreyRectangle = False
rebalanceToggle=False
localRebalanceToggle=False
ListDataToggle=False
RecordFlag=False
GreyRect = (0,0,1,1) 
times=[]
clocks=[]
hues=[]
huesr=[]
HSVMeans=[]
RGBMeans=[]
LABMeans=[]
RGBGreys=[]
reds=[]
greens=[]
blues=[]
LABa=[]
LABb=[]
bWB=[]
gWB=[]
rWB=[]
dataPoint=0
interlaceStart=0
RectList=[]
#RectList=[(211, 149, 43, 132), (274, 151, 49, 129), (344, 153, 46, 127)]
#RectList=[(224, 154, 59, 153), (283, 161, 55, 145), (338, 149, 59, 155)]
#RectList=[(214, 135, 52, 153), (270, 140, 63, 151), (338, 139, 65, 148)]
#ColorRectOver=True
GreyRectOver = False
font = cv2.FONT_HERSHEY_SIMPLEX
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
file_base='C:/Users/Kevin/Dropbox/Whatman/'
#RectList=[(212, 151, 75, 163)]


def onmouse(event,x,y,flags,params):
    global ColorRectangle,ColorRect,ix,iy,ixg,iyg,ColorRectOver,GreyRectangle,GreyRect,GreyRectOver,rebalanceToggle,RectList
    if event == cv2.EVENT_LBUTTONDOWN:
            ColorRectangle = True
            ColorRectOver = False
            ix,iy = x,y
    elif event == cv2.EVENT_RBUTTONDOWN:
        if GreyRectOver==True:
            GreyRectOver=False
        else:
            GreyRectangle = True
            GreyRectOver = False
            ixg,iyg = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if ColorRectangle == True:
            cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),2)
            ColorRect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            cv2.imshow('Result',img)
            #cv2.waitKey(1)
        if GreyRectangle == True:
            cv2.rectangle(img,(ixg,iyg),(x,y),(0,0,0),2)
            GreyRect = (min(ixg,x),min(iyg,y),abs(ixg-x),abs(iyg-y))
            cv2.imshow('Result',img)
            #cv2.waitKey(1)
    elif event == cv2.EVENT_LBUTTONUP:
        ColorRectangle = False
        if ix!=x & iy!=y:
            ColorRectOver = True
            cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),2)
            ColorRect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            x1,y1,w,h = ColorRect
            RectList.append(ColorRect)
            cv2.imshow('Result',img)
        else:
            ColorRectOver = False
    elif event == cv2.EVENT_RBUTTONUP:
        GreyRectangle = False
        if ixg!=x & iyg!=y:
            GreyRectOver = True
            cv2.rectangle(img,(ixg,iyg),(x,y),(0,0,0),2)
            GreyRect = (min(ixg,x),min(iyg,y),abs(ixg-x),abs(iyg-y))
            #x1,y1,w,h = GreyRect        
            #cv2.imshow('Result',img)
            rebalanceToggle=True
        else:
            GreyRectOver = False
            
def OpenCVDisplayedHistogram(image,channel,mask,NumBins,DataMin,DataMax,x,y,w,h,DisplayImage,color,integrationWindow,labelFlag):
        avgVal=cv2.meanStdDev(image,mask=mask)
        histdata = cv2.calcHist([image],[channel],mask,[NumBins],[DataMin,DataMax])
        domValue=np.argmax(histdata)
        numpixels=sum(np.array(histdata[domValue-integrationWindow:domValue+integrationWindow+1]))
        cv2.normalize(histdata, histdata, 0, h, cv2.NORM_MINMAX);
        if w>NumBins:
            binWidth = w/NumBins
        else:
            binWidth=1
        #img = np.zeros((h, NumBins*binWidth, 3), np.uint8)
        for i in range(NumBins):
            freq = int(histdata[i])
            cv2.rectangle(DisplayImage, ((i*binWidth)+x, y+h), (((i+1)*binWidth)+x, y+h-freq), color)
        if labelFlag:
            cv2.putText(img,"m="+'{0:.2f}'.format(domValue/float(NumBins)*(DataMax-DataMin))+" a="+'{0:.2f}'.format(avgVal[0][channel][0])+" s="+'{0:.2f}'.format(avgVal[1][channel][0]),(x,y+h+20), font, 0.5,color,1,cv2.LINE_AA)
        return (avgVal[0][channel][0],avgVal[1][channel][0])
    
    
#    def OpenCVDisplayedHistogram(image,channel,mask,NumBins,DataMin,DataMax,x,y,w,h,DisplayImage,color,integrationWindow,labelFlag):
#        avgVal=cv2.mean(image,mask=mask)[channel]
#        pix=np.sum(mask)/255
#        histdata = cv2.calcHist([image],[channel],mask,[NumBins],[DataMin,DataMax])
#        domValue=np.argmax(histdata)
#        numpixels=sum(np.array(histdata[domValue-integrationWindow:domValue+integrationWindow+1]))
#        cv2.normalize(histdata, histdata, 0, h, cv2.NORM_MINMAX);
#        binWidth = w/NumBins
#        for i in xrange(NumBins):
#            freq = int(histdata[i])
#            cv2.rectangle(DisplayImage, ((i*binWidth)+x, y+h), (((i+1)*binWidth)+x, y+h-freq), color)
#        if labelFlag:
#            cv2.putText(img,"avg="+'{0:.3f}'.format(avgVal)+" pix="+str(pix).zfill(4),(x,y+h+20), font, 0.6,color,1,cv2.LINE_AA)
#        return (domValue/float(NumBins),numpixels)

def boxcarSmooth(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')
        
def OpenCVDisplayedScatter(xdata,ydata,x,y,w,h,size,color,ydataRangemin=None, ydataRangemax=None,xdataRangemin=None, xdataRangemax=None):      
        if xdataRangemin==None: 
             xdataRangemin=min(xdata)       
        if xdataRangemax==None: 
             xdataRangemax=max(xdata) 
        if ydataRangemin==None: 
             ydataRangemin=min(ydata) 
        if ydataRangemax==None: 
             ydataRangemax=max(ydata)
        xdataRange=xdataRangemax-xdataRangemin
        ydataRange=ydataRangemax-ydataRangemin
        xscale=w/xdataRange
        yscale=h/ydataRange
        xdata=xdata-xdataRangemin
        ydata=ydata-ydataRangemin
        xdata=(xdata*xscale).astype(int)
        ydata=(ydata*yscale).astype(int)
        for i in range(xdata.size):
            cv2.circle(img,( xdata[i]+x,h-ydata[i]+y ), size , color, -1)
#        for i in range(xdata.size-1):
#            cv2.line(img,( xdata[i]+x,h-ydata[i]+y ), ( xdata[i+1]+x,h-ydata[i+1]+y ), color, size )
        cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
        cv2.putText(img,str(round(xdataRangemax,0)),(x+w-15,y+h+15), font, 0.5,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(xdataRangemin,0)),(x-5,y+h+15), font, 0.5,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(ydataRangemax,2)),(x-40,y+10), font, 0.5,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(ydataRangemin,2)),(x-40,y+h-5), font, 0.5,color,1,cv2.LINE_AA)
#x=array([0,1.0,2.0,3.0,4.0,5.0])
#y=array([.1,11,23,32,39,50])
#pts=np.transpose(np.vstack((x, y)))
#vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, .01, .01)
#slope=(cy-vy)/(cx-vx)
#intercept=cy-(slope*cx)
#src1=np.transpose(np.vstack((ones_like(x), x)))
#src2=y
#coeffs=cv2.solve(src1,src2,flags=cv2.DECOMP_SVD)
#slp=coeffs[1][1]
#inter=coeffs[1][0]
        
def RebalanceImageCV(frame,rfactor,gfactor,bfactor):
    #could I scale up to a 16-bit image here?
    offset=np.zeros(frame[:,:,0].shape,dtype="uint8")
    frame[:,:,0]=cv2.scaleAdd(frame[:,:,0], bfactor, offset)
    frame[:,:,1]=cv2.scaleAdd(frame[:,:,1], gfactor, offset)
    frame[:,:,2]=cv2.scaleAdd(frame[:,:,2], rfactor, offset)
    return frame

def SubsampleData(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)
#    cv.Reshape(arr, newCn, newRows=0)
    
#cap = cv2.VideoCapture("v2.avi")
#cap = cv2.VideoCapture("Video 1.mp4")
#cap = cv2.VideoCapture(0)

#for filenum in range(45):
#    file_path=file_base+str(filenum+1).zfill(2)+".tif"
root = tk.Tk()
root.withdraw()
file_path = askopenfilename()
if len(file_path)==0:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(file_path)
if RecordFlag:
    outp = cv2.VideoWriter(file_path+"Processed Titration 12_3_15.avi",fourcc, 30.0, (1700, 900))
    #outr = cv2.VideoWriter("Recorded Titration 12_3_15.avi",fourcc, 30.0, (640, 480))


cv2.namedWindow('Result',cv2.WINDOW_GUI_NORMAL)
#cv2.namedWindow('Result')
cv2.setMouseCallback('Result',onmouse)
FrameRate= cap.get(cv2.CAP_PROP_FPS)
TotalFrames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
TotalFrames=200000
HueMatrixSize=int(math.ceil(TotalFrames/MeanWindow))
#pre-allocate a numpy array here to and store the list values in it for speed
startTime = time.time()
HueAvgs=np.zeros((HueMatrixSize,len(RectList)))
AreaAvgs=np.zeros((HueMatrixSize,len(RectList)))
LABaAvgs=np.zeros((HueMatrixSize,len(RectList)))
LABbAvgs=np.zeros((HueMatrixSize,len(RectList)))
        
#while(cap.get(cv2.CAP_PROP_POS_FRAMES)<TotalFrames-1):
#    ret, frame = cap.read()
frame16=cv2.imread(file_path,cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
if frame16.dtype=='uint16':
    frame=np.uint8(frame16*1./255)
else:
    frame=frame16

while(True):
#    ret, frame = cap.read()

    imgScale=50.0/max(frame.shape[1]/16.0,frame.shape[0]/9.0)
    img = np.zeros((900, 1700, 3), np.uint8)
    CamFrame = cv2.resize(frame, (int(frame.shape[1]*imgScale),int(frame.shape[0]*imgScale)), interpolation = cv2.INTER_AREA)
    img[0:int(frame.shape[0]*imgScale),0:int(frame.shape[1]*imgScale),:]=CamFrame

    if FrameRate!=0:
        vidTime=(cap.get(cv2.CAP_PROP_POS_FRAMES)-1)/FrameRate
    else:
        vidTime = time.time()-startTime
    realTime = time.time()-startTime
    if ColorRectangle:
        x1,y1,w,h = ColorRect   
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,255,255),2)
    if GreyRectangle:
        x1,y1,w,h = GreyRect   
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,0,0),2)
    if GreyRectOver:
        x1,y1,w,h = GreyRect
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,0,0),2)
        x1=int(x1/imgScale)
        y1=int(y1/imgScale)
        w=int(w/imgScale)
        h=int(h/imgScale)
        GreyROI=frame[y1:y1+h, x1:x1+w]
        RGBGreyROI=cv2.mean(GreyROI)
        bscale=RGBGreyROI[0]
        gscale=RGBGreyROI[1]
        rscale=RGBGreyROI[2]
        scalemax=max(rscale,gscale,bscale)
        scalemin=min(rscale,gscale,bscale)
        if scalemin!=0:
            rfactor=float(scalemin)/float(rscale)
            gfactor=float(scalemin)/float(gscale)
            bfactor=float(scalemin)/float(bscale)
    else:
        rfactor=float(1)
        gfactor=float(1)
        bfactor=float(1)
    if rebalanceToggle:
        RebalanceImageCV(frame,rfactor,gfactor,bfactor)

    if ColorRectOver:
        recNum=0
        for ColorRect in RectList:    
            x1,y1,w,h = ColorRect
            cv2.rectangle(img,(int(frame.shape[1]*imgScale)+(recNum*260),0),(int(frame.shape[1]*imgScale)+(recNum*260)+260,900),(255-(recNum*75),255-(recNum*75),255-(recNum*75)),2)
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255-(recNum*75),255-(recNum*75),255-(recNum*75)),2)
            cv2.rectangle(img,(x1,y1+int(frame.shape[0]*imgScale)),(x1+w,y1+h+int(frame.shape[0]*imgScale)),(255-(recNum*75),255-(recNum*75),255-(recNum*75)),2)
            img[y1+450:y1+450+h,x1:x1+w,:]= CamFrame[y1:y1+h,x1:x1+w,:]
            x1f=int(x1/imgScale)
            y1f=int(y1/imgScale)
            wf=int(w/imgScale)
            hf=int(h/imgScale)
            rgbROI = frame[y1f:y1f+hf, x1f:x1f+wf]
            if localRebalanceToggle:
                hsvROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2HSV)
                maskROI = cv2.inRange(hsvROI, lower_lim, upper_lim)
                maskGreyROI = cv2.bitwise_not(maskROI)
                RGBGrey=cv2.mean(rgbROI, mask=maskGreyROI)
                Bw=RGBGrey[0]
                Gw=RGBGrey[1]
                Rw=RGBGrey[2]
                Rr=max([Bw,Gw,Rw])/Rw
                Gr=max([Bw,Gw,Rw])/Gw
                Br=max([Bw,Gw,Rw])/Bw
                rgbROI=RebalanceImageCV(rgbROI,Rr,Gr,Br)
            hsvROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2HSV)
            labROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2Lab)
            absROI=np.float32(-np.log10(rgbROI/255.0))
            #absROI=np.float32(-np.log10(rgbROI/255.0)/-np.log10(1/255.0))
            absROI[absROI>2]=2
            #absDifROI=np.float32(-np.log10((rgbROI[:,:,2]-rgbROI[:,:,0])/255.0)/-np.log10(1/255.0))
            maskROI = cv2.inRange(hsvROI, lower_lim, upper_lim)
            maskGreyROI = cv2.bitwise_not(maskROI)
            resFrame = cv2.bitwise_and(rgbROI,rgbROI, mask= maskROI)
            CamMaskFrame = cv2.resize(resFrame, (w,h), interpolation = cv2.INTER_AREA)
            img[y1+450:y1+450+h,x1:x1+w,:]= CamMaskFrame
            RGBGrey=cv2.mean(rgbROI, mask=maskGreyROI)
            HSVMean=cv2.mean(hsvROI, mask=maskROI)
            RGBMean=cv2.mean(rgbROI, mask=maskROI)
            LABMean=cv2.mean(labROI, mask=maskROI)
            row=0                
            aH,sH=OpenCVDisplayedHistogram(hsvROI,0,maskROI,180,0,180,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),180,40,img,(255,255,0),1,True)
            row+=1
            aS,sS=OpenCVDisplayedHistogram(hsvROI,1,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(200,200,200),1,True)
            row+=1
            aV,sS=OpenCVDisplayedHistogram(hsvROI,2,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(128,128,128),1,True)
            row+=1
            aL,sL=OpenCVDisplayedHistogram(labROI,0,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,255,255),1,True)
            row+=1
            aa,sa=OpenCVDisplayedHistogram(labROI,1,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,0,255),1,True)
            row+=1
            ab,sb=OpenCVDisplayedHistogram(labROI,2,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,255,255),1,True)
            row+=1
            aB,sB=OpenCVDisplayedHistogram(rgbROI,0,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,0,0),5,True)
            #Bw,numBw=OpenCVDisplayedHistogram(rgbROI,0,maskGreyROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(128,0,0),5,False)
            row+=1
            aG,sG=OpenCVDisplayedHistogram(rgbROI,1,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,255,0),5,True)
            #Gw,numGw=OpenCVDisplayedHistogram(rgbROI,1,maskGreyROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,128,0),5,False)
            row+=1
            aR,sR=OpenCVDisplayedHistogram(rgbROI,2,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,0,255),5,True)
            #Rw,numRw=OpenCVDisplayedHistogram(rgbROI,2,maskGreyROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,0,128),5,False)
            row+=1
            aGoR,sGoR=OpenCVDisplayedHistogram(np.float32(rgbROI[:,:,1].astype(float)/rgbROI[:,:,2]),0,maskROI,256,0,1,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,0,0),5,True)
            row+=1
            aBoR,sBoR=OpenCVDisplayedHistogram(np.float32(rgbROI[:,:,0].astype(float)/rgbROI[:,:,2]),0,maskROI,256,0,1,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,255,0),5,True)
            row+=1
            aRoG,sRoG=OpenCVDisplayedHistogram(np.float32(rgbROI[:,:,0].astype(float)/rgbROI[:,:,1]),0,maskROI,256,0,1,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,0,255),5,True)
            row+=1
            aRmG,sRmG=OpenCVDisplayedHistogram(rgbROI[:,:,2]-rgbROI[:,:,1],0,maskROI,256,0,256,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,255,255),5,True)
            row+=1
            aRmG,sRmG=OpenCVDisplayedHistogram(rgbROI[:,:,2]-rgbROI[:,:,0],0,maskROI,256,0,256,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,0,255),5,True)
            row+=1
            aRmG,sRmG=OpenCVDisplayedHistogram(rgbROI[:,:,1]-rgbROI[:,:,0],0,maskROI,256,0,256,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,255,0),5,True)
            row+=1
#            #nitrate
#            agr,numR=OpenCVDisplayedHistogram(absROI[:,:,1]-absROI[:,:,2],0,maskROI,256,0,.2,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(0,255,255),5,True)
#            #ophen
#            #ad,numR=OpenCVDisplayedHistogram(absROI[:,:,0]-absROI[:,:,2],0,maskROI,256,0,1,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,0,255),5,True)
#            #bradford            
#            #ad,numR=OpenCVDisplayedHistogram(absROI[:,:,2]/absROI[:,:,0],0,maskROI,256,0,1,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,0,255),5,True)
#            row+=1
#            abr,numR=OpenCVDisplayedHistogram(absROI[:,:,0]-absROI[:,:,2],0,maskROI,256,0,.2,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,0,255),5,True)
#            row+=1
#            abg,numR=OpenCVDisplayedHistogram(absROI[:,:,0]-absROI[:,:,1],0,maskROI,256,0,.2,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,255,0),5,True)
#            row+=1
#            #adi,numR=OpenCVDisplayedHistogram(rgbROI[:,:,2]-rgbROI[:,:,0],0,maskROI,256,0,255,int(frame.shape[1]*imgScale)+(recNum*260),5+(row*55),256,40,img,(255,255,255),5,True)
#            #OpenCVDisplayedHistogram(image,channel,mask,NumBins,DataMin,DataMax,x,y,w,h,DisplayImage,color,integrationWindow,labelFlag):


            recNum=recNum+1

#    OpenCVDisplayedScatter(distance[:,pad],CalibrationData[:,pad,0],CamFrame.shape[1]+100+(pad*300),750,100,100,1,(255,255,255),0.0,14.0)

        
    if RecordFlag:
        cv2.putText(img,"REC",(10,10), font, .5,(0,0,255),2,cv2.LINE_AA)  
        outp.write(img)
        #outr.write(frame)
    cv2.imshow('Result', img)        
    if ListDataToggle==False:
        times=[]
        clocks=[]
        hues=[]
        HSVMeans=[]
        RGBMeans=[]
        LABMeans=[]
        RGBGreys=[]
        huesr=[]
        reds=[]
        greens=[]
        blues=[]
        LABa=[]
        LABb=[]
        sats=[]
        vals=[]
        sizes=[]
        areas=[]
        rectAreas=[]
        bWB=[]
        gWB=[]
        rWB=[]
        dataPoint=0
        interlaceStart=0
    if FrameByFrameToggle:
        keypress=cv2.waitKey(0) & 0xFF
    else:
        keypress=cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        break
    if keypress == ord(' '):
        ColorRectOver = False
        GreyRectOver=False
        RectList=[]
        ListDataToggle=False
    if keypress == ord('n'):
        ColorRectOver = False
        RectList=[]
        ListDataToggle=False
    if keypress == ord('w'):
        rebalanceToggle=not rebalanceToggle
    if keypress == ord('c'):
        localRebalanceToggle=not localRebalanceToggle
    if keypress == ord('l'):
        ListDataToggle=not ListDataToggle
        if ListDataToggle==True:
            HueAvgs=np.zeros((HueMatrixSize,len(RectList)))
            AreaAvgs=np.zeros((HueMatrixSize,len(RectList)))
            LABaAvgs=np.zeros((HueMatrixSize,len(RectList)))
            LABbAvgs=np.zeros((HueMatrixSize,len(RectList)))
    if keypress == ord('f'):
        FrameByFrameToggle=not FrameByFrameToggle  
    if keypress == ord('r'):
        RecordFlag=not RecordFlag
cap.release()
#cv2.imwrite(file_path+"ProcessedAll.jpg",img)
#outp.release()
#outr.release()
cv2.destroyAllWindows()

