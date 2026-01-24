import pandas as pd
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

#startpH=2.0
#incrementpH=0.5
#numPHs=8
#numPads=4
numWaves=1560

#C:\Users\Public\Dropbox\Su15 Color\OOData\05_31_16\Convert
#OODirectory='C:\Users\Kevin\Dropbox\Su15 Color\OOData\Convert'
#OODirectory=r'C:\Users\Public\Dropbox\Su15 Color\OOData\\05_31_16\Convert'
#OODirectory=r'C:\Users\Kevin\Dropbox\Su15 Color\OOData\\05_31_16\Convert'
OODirectory=r'/home/kevin/UofP Dropbox/Kevin Cantrell/Su15 Color/OOData/05_31_16/Convert'
OOFileNameA=r'/WhatWide'
OOFileNameB='_pH'
OOFileNameC='_'
OOFileNameD='_pad'
OOFileNameE='_trial'
OOFileNameF='.txt'

#pHValues=[11.0,10.5,10.0,9.5,9.0,8.5,8.0,7.5,7.0,6.5,6.0,5.5,5.0,4.5,4.0]
pHValues=[14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0]
padValues=[1,2,3,4]
trialValues=[1,2,3]
XYZtolRGB=np.array([[3.2406255,-1.537208,-0.4986286],[-0.9689307,1.8757561,0.0415175],[0.0557101,-0.2040211,1.0569959]])


def  XYZtransferLAB(t):
    if t>(6/29.0)**3:
        return(t**(1/3.0))
    else:
        return(((1/3.0*(29/6.0)**2)*t)+4/29.0)

def interpolateResponse(OOWaves, CIEWaves, X):
    numCIEwaves=CIEWaves.shape[0]
    Xind=np.zeros((numCIEwaves+int(math.floor(OOWaves[0]))))
    for index in range(numCIEwaves):
        newindex=int(CIEWaves[index])
        Xind[newindex]=X[index]
    numOOwaves=OOWaves.shape[0]
    Xout=np.zeros((numOOwaves))
    for i in range(numOOwaves):
        roundedlow=int(math.floor(OOWaves[i]))
        roundedhigh=roundedlow+1
        if roundedhigh<CIEWaves[numCIEwaves-1]:
            Xout[i]=((Xind[roundedhigh]-Xind[roundedlow])*(OOWaves[i]-roundedlow))+Xind[roundedlow]
    return (Xout)

ReflectanceArray=np.zeros((len(pHValues),len(padValues),len(trialValues),numWaves))
WavelengthArray=np.zeros((numWaves))
pHs=np.zeros((len(pHValues)))
pHIndex=0
for pHValue in pHValues:
    pHs[pHIndex]=pHValue
    padIndex=0
    for pad in padValues:
        trialIndex=0
        for trial in trialValues:
            OOFileName= OOFileNameA+OOFileNameB+str(pHValue).split('.')[0]+OOFileNameC+str(pHValue).split('.')[1]+OOFileNameD+str(pad)+OOFileNameE+str(trial)+OOFileNameF
            OOFileDF=pd.read_table(OODirectory+OOFileName,skiprows=505,skipfooter=1,header=None, engine='python')
            ReflectanceArray[pHIndex,padIndex,trialIndex,:]=OOFileDF.values[:,1]
            trialIndex=trialIndex+1
        padIndex=padIndex+1
    pHIndex=pHIndex+1
WavelengthArray=OOFileDF.values[:,0]

refRange=((WavelengthArray>800) & (WavelengthArray<850))
TargetReflectance=75
if TargetReflectance!=0:
    CorrectedReflectanceArray=np.copy(ReflectanceArray)
    TrialFig,CorrectedAxes=plt.subplots(2,len(padValues),sharex=True,sharey=True)
    for pad in range(len(padValues)):
        for pH in range(len(pHValues)):
            for trial in range(len(trialValues)):
                CF=np.mean(ReflectanceArray[pH,pad,trial,refRange])/TargetReflectance
                CorrectedReflectanceArray[pH,pad,trial,:]=ReflectanceArray[pH,pad,trial,:]/CF
                CorrectedAxes[0,pad].plot(WavelengthArray,ReflectanceArray[pH,pad,trial,:],label="ph"+str(pHs[pH])+' tr'+str(trialValues[trial]))
                CorrectedAxes[1,pad].plot(WavelengthArray,CorrectedReflectanceArray[pH,pad,trial,:],label="ph"+str(pHs[pH])+' tr'+str(trialValues[trial]))
    ReflectanceArray=CorrectedReflectanceArray
    
AbsorbanceArray=-np.log10(ReflectanceArray/100.0)


InColorArray=np.zeros((15,5,3))
#EXFileName= r'C:\Users\Public\Dropbox\Su15 Color\WhatmanValuesRef2.csv'
EXFileName= r'/home/kevin/UofP Dropbox/Kevin Cantrell/Su15 Color/WhatmanValuesRef2.csv'
EXFileDF=pd.read_csv(EXFileName)
for pad in range(5):
    for pHValue in range(15):
        InColorArray[pHValue,pad,:]=EXFileDF.values[(pad*15)+pHValue,2:5]

#PhotoColorArray=np.zeros((15,3,3))
#EXFileName= 'C:\Users\Public\Dropbox\Su15 Color\Photos\\06_07_2016-317e.tifRGBoutput.csv'
#EXFileDF=pd.read_csv(EXFileName)
#for pad in range(3):
#    for pHValue in range(15):
#        PhotoColorArray[pHValue,pad,:]=EXFileDF.values[(pad*15)+pHValue,1:4]

#XLFileDF = pd.read_excel('C:\Users\Public\Dropbox\Su15 Color\OOData\\all_1nm_data.xls',skiprows=63)
XLFileDF = pd.read_excel(r'~/UofP Dropbox/Kevin Cantrell/Su15 Color/OOData/all_1nm_data.xls',skiprows=63)
CIEX=interpolateResponse(WavelengthArray, XLFileDF.values[:,0],XLFileDF.values[:,5])
CIEY=interpolateResponse(WavelengthArray, XLFileDF.values[:,0],XLFileDF.values[:,6])
CIEZ=interpolateResponse(WavelengthArray, XLFileDF.values[:,0],XLFileDF.values[:,7])
D65=interpolateResponse(WavelengthArray, XLFileDF.values[:,0],XLFileDF.values[:,2])

Xr=np.trapezoid(CIEX*D65, WavelengthArray)
Yr=np.trapezoid(CIEY*D65, WavelengthArray)
Zr=np.trapezoid(CIEZ*D65, WavelengthArray)

Xn=np.trapezoid(CIEX*D65*(1), WavelengthArray)/Yr
Yn=np.trapezoid(CIEY*D65*(1), WavelengthArray)/Yr
Zn=np.trapezoid(CIEZ*D65*(1), WavelengthArray)/Yr

ColorArray=np.zeros((len(pHValues),len(padValues),6))
for pH in range(len(pHValues)):
    for pad in range(len(padValues)):
        X=np.trapezoid(CIEX*D65*(np.mean(ReflectanceArray[pH,pad,:,:],axis=0)/100), WavelengthArray)/Yr
        Y=np.trapezoid(CIEY*D65*(np.mean(ReflectanceArray[pH,pad,:,:],axis=0)/100), WavelengthArray)/Yr
        Z=np.trapezoid(CIEZ*D65*(np.mean(ReflectanceArray[pH,pad,:,:],axis=0)/100), WavelengthArray)/Yr
        L=116*XYZtransferLAB(Y/Yn)-16
        A=500*(XYZtransferLAB(X/Xn)-XYZtransferLAB(Y/Yn))
        B=200*(XYZtransferLAB(Y/Yn)-XYZtransferLAB(Z/Zn))
        RGBl=np.dot(XYZtolRGB,[X,Y,Z])
        RGBg=np.zeros((RGBl.shape))
        RGBs=np.zeros((RGBl.shape))
        for cc in range(RGBl.shape[0]):
            if RGBl[cc]<=0.0031308:
                RGBg[cc]=12.92*RGBl[cc]
            else:
                RGBg[cc]=1.055*RGBl[cc]**(1/2.4)-0.055
            RGBs[cc]=int(round(RGBg[cc]*255.0))
            if RGBs[cc]>255:
                RGBs[cc]=255
            elif RGBs[cc]<0:
                RGBs[cc]=0
        ColorArray[pH,pad,0:3]=RGBs
        ColorArray[pH,pad,3]=L
        ColorArray[pH,pad,4]=A
        ColorArray[pH,pad,5]=B
        
stripinc=92
padinc=100
stripstart=256
padstart=260
circler=40

ReferenceImage = np.full((1200, 1800, 3), 255,np.uint8)
for pH in range(len(pHValues)):
    for pad in range(len(padValues)):
        cv2.circle(ReferenceImage,((stripinc*pH)+stripstart,(padinc*pad)+padstart), circler, (ColorArray[pH,pad,2],ColorArray[pH,pad,1],ColorArray[pH,pad,0]), -1)
cv2.circle(ReferenceImage,(155,155), circler*2, (0,255,255), -1)
cv2.circle(ReferenceImage,(1645,155), circler*2, (0,255,0), -1)
cv2.circle(ReferenceImage,(1645,1045), circler*2, (255,255,0), -1)
cv2.circle(ReferenceImage,(155,1045), circler*2, (255,0,0), -1)
cv2.rectangle(ReferenceImage,(0,0), (1800,1200), (255,0,255), 120)
cv2.rectangle(ReferenceImage,(370,770), (1382,893), (255,0,255), 50)
cv2.rectangle(ReferenceImage,(415,790), (475,873), (0,0,0), -1)
cv2.imshow('RefCard', ReferenceImage)
#cv2.imwrite(OODirectory+OOFileNameA+"RefCard.jpg", ReferenceImage)

fig,axes=plt.subplots(2,len(padValues),sharex=True,sharey=False)
for pad in range(len(padValues)):
    for pH in range(len(pHValues)):
#        axes[0,pad].plot(WavelengthArray,np.mean(ReflectanceArray[pH,pad,:,:],axis=0),label="ph"+str(pHs[pH]),color=(ColorArray[pH,pad,0:3]/255.))
#        axes[1,pad].plot(WavelengthArray,np.mean(AbsorbanceArray[pH,pad,:,:],axis=0),label="ph"+str(pHs[pH]),color=(ColorArray[pH,pad,0:3]/255.))        
        axes[0,pad].plot(WavelengthArray,np.mean(ReflectanceArray[pH,pad,:,:],axis=0),label="ph"+str(pHs[pH]))
        axes[1,pad].plot(WavelengthArray,np.mean(AbsorbanceArray[pH,pad,:,:],axis=0),label="ph"+str(pHs[pH]))        
#        for trial in range(len(trialValues)):
#            axes[0,pad].plot(WavelengthArray,ReflectanceArray[pH,pad,trial,:],label="ph"+str(pHs[pH])+' tr'+str(trial+1))
#            axes[1,pad].plot(WavelengthArray,AbsorbanceArray[pH,pad,trial,:],label="ph"+str(pHs[pH])+' tr'+str(trial+1))
#    axes[0,pad].legend(loc='best')
#    axes[1,pad].legend(loc='best')
    axes[0,pad].set_xlim([360, 880])
    axes[0,pad].set_ylim([0, 120])
    axes[1,pad].set_xlim([360, 880])
    axes[1,pad].set_ylim([0, 2])    
    
TrialFig,TrialAxes=plt.subplots(len(pHValues),len(padValues),sharex=True,sharey=True)
for pad in range(len(padValues)):
    for pH in range(len(pHValues)):
#        TrialAxes[pH,pad].plot(WavelengthArray,np.mean(ReflectanceArray[pH,pad,:,:],axis=0),label="ph"+str(pHs[pH])+' avg',color=(ColorArray[pH,pad,0:3]/255.))
        for trial in range(len(trialValues)):
            TrialAxes[pH,pad].plot(WavelengthArray,ReflectanceArray[pH,pad,trial,:],label="ph"+str(pHs[pH])+' tr'+str(trialValues[trial]))
    TrialAxes[0,pad].set_xlim([360, 880])
    TrialAxes[0,pad].set_ylim([0, 120])

TrialFig,TrialAxes=plt.subplots(len(pHValues),len(padValues),sharex=True,sharey=True)
for pad in range(len(padValues)):
    for pH in range(len(pHValues)):
#        TrialAxes[pH,pad].plot(WavelengthArray,np.mean(AbsorbanceArray[pH,pad,:,:],axis=0),label="ph"+str(pHs[pH])+' avg',color=(ColorArray[pH,pad,0:3]/255.))
        for trial in range(len(trialValues)):
            TrialAxes[pH,pad].plot(WavelengthArray,AbsorbanceArray[pH,pad,trial,:],label="ph"+str(pHs[pH])+' tr'+str(trialValues[trial]))
    TrialAxes[0,pad].set_xlim([360, 880])
    TrialAxes[0,pad].set_ylim([0, 2])
    
CCFig,CCaxes=plt.subplots(3,len(padValues),sharex=True,sharey=True)
for pad in range(len(padValues)):
        CCaxes[0,pad].plot(pHs,ColorArray[:,pad,0],'-or',label=str(pHs[pH]))
        CCaxes[0,pad].plot(pHs,ColorArray[:,pad,1],'-og')
        CCaxes[0,pad].plot(pHs,ColorArray[:,pad,2],'-ob')
        CCaxes[1,pad].plot([11.0,10.5,10.0,9.5,9.0,8.5,8.0,7.5,7.0,6.5,6.0,5.5,5.0,4.5,4.0],InColorArray[:,pad,0],'-or')
        CCaxes[1,pad].plot([11.0,10.5,10.0,9.5,9.0,8.5,8.0,7.5,7.0,6.5,6.0,5.5,5.0,4.5,4.0],InColorArray[:,pad,1],'-og')
        CCaxes[1,pad].plot([11.0,10.5,10.0,9.5,9.0,8.5,8.0,7.5,7.0,6.5,6.0,5.5,5.0,4.5,4.0],InColorArray[:,pad,2],'-ob')
#        CCaxes[2,pad].plot([11.0,10.5,10.0,9.5,9.0,8.5,8.0,7.5,7.0,6.5,6.0,5.5,5.0,4.5,4.0],PhotoColorArray[:,pad,0],'-or')
#        CCaxes[2,pad].plot([11.0,10.5,10.0,9.5,9.0,8.5,8.0,7.5,7.0,6.5,6.0,5.5,5.0,4.5,4.0],PhotoColorArray[:,pad,1],'-og')
#        CCaxes[2,pad].plot([11.0,10.5,10.0,9.5,9.0,8.5,8.0,7.5,7.0,6.5,6.0,5.5,5.0,4.5,4.0],PhotoColorArray[:,pad,2],'-ob')
        CCaxes[0,pad].set_xlim([0, 14])
        CCaxes[0,pad].set_ylim([0, 255])
        CCaxes[1,pad].set_xlim([0, 14])
        CCaxes[1,pad].set_ylim([0, 255]) 
        CCaxes[2,pad].set_xlim([0, 14])
        CCaxes[2,pad].set_ylim([0, 255]) 

LabFig,Labaxes=plt.subplots(3,len(padValues),sharex=True,sharey=True)
for pad in range(len(padValues)):
        Labaxes[0,pad].plot(pHs,ColorArray[:,pad,3],'-ok',label=str(pHs[pH]))
        Labaxes[0,pad].plot(pHs,ColorArray[:,pad,4],'-om')
        Labaxes[0,pad].plot(pHs,ColorArray[:,pad,5],'-oy')
