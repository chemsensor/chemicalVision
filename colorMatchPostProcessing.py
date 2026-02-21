import matplotlib.pyplot as plt
import numpy as np

def EuclidianDistance(cc1,cc2):
    coordinates=len(cc1)
    totalDistance=0
    for axis in range(coordinates):
        distance=cc1[axis]-cc2[axis]
        totalDistance=totalDistance+distance**2
    totatalDistance=np.sqrt(totalDistance)
    return totatalDistance


#only using a and b of the Lab color space    
for startIndex,endIndex in zip([0,60],[60,124]):       
    pads=standardSwatchStats[14,0,0,startIndex:endIndex]
    pHs=standardSwatchStats[13,0,0,startIndex:endIndex]
    padList=set(pads)
    samLAB=parameterStats[7:9,0,0,0:4]
    fig,axes=plt.subplots(len(padList),1,sharex=True,sharey=True)
    cc=7
    for padNumber in padList:
        padMask=padNumber==standardSwatchStats[14,0,0,startIndex:endIndex]
        axes[int(padNumber)].plot(pHs[padMask],standardSwatchStats[cc,0,0,startIndex:endIndex][padMask],'-ok')
        axes[int(padNumber)].plot(pHs[padMask],standardSwatchStats[cc+1,0,0,startIndex:endIndex][padMask],'-om')
        #axes[int(padNumber)].plot(pHs[padMask],standardSwatchStats[cc+2,0,0,startIndex:endIndex][padMask],'-oy')
        axes[int(padNumber)].plot([0,14],[samLAB[0,int(padNumber)],samLAB[0,int(padNumber)]],':k')
        axes[int(padNumber)].plot([0,14],[samLAB[1,int(padNumber)],samLAB[1,int(padNumber)]],':m')
        #axes[int(padNumber)].plot([0,14],[samLAB[2,int(padNumber)],samLAB[2,int(padNumber)]],':y')
    
    
    #for LAB color space initial cc is 6
    refLAB=standardSwatchStats[7:9,0,0,startIndex:endIndex]
    samLAB=parameterStats[7:9,0,0,0:4]
    pads=standardSwatchStats[14,0,0,startIndex:endIndex]
    numPads=samLAB.shape[1]
    numRefsPads=refLAB.shape[1]
    numRefs=int(numRefsPads/numPads)
    distances=np.zeros((numRefs))
    pHrefs=np.zeros((numRefs))
    for pad in range(numPads):
        cc1=samLAB[:,pad]
        padMask=pad==pads
        refSamePad=refLAB[:,padMask]
        for ref in range(numRefs):
            cc2=refSamePad[:,ref]
            distances[ref]=distances[ref]+EuclidianDistance(cc1,cc2)
            pHrefs[ref]=pHs[padMask][ref]
    fig,ax=plt.subplots()
    ax.plot(pHrefs,distances)
    
    closestIndex=np.argmin(distances)
    closest_pH=pHrefs[closestIndex]
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import numpy as np

channel=1
EuclidDist=np.zeros((4,100,2))
intPh=np.zeros((4,100,2))
DAg=np.zeros((100,2))
Dstk=np.zeros((200))
if (ClosestIndex-1)>=0 and (ClosestIndex+1)<=14:
    for std in range(2):
        for travel in range(100):
            for pad in range(4):
                    X0=UnknownData[pad,channel:channel+3]
                    X1=CalibrationData[ClosestIndex-1+std,pad,channel:channel+3]
                    X2=CalibrationData[ClosestIndex+std,pad,channel:channel+3]
                    #tu=-numpy.dot(X1-X0,X2-X1)/numpy.absolute(numpy.dot(X2-X1,X2-X1))
                    tu=travel/100.
                    X3=X1+((X2-X1)*tu)
                    EuclidDist[pad,travel,std]=numpy.linalg.norm(X3-X0)
                    intPh[pad,travel,std]=CalibrationData[ClosestIndex-1+std,0,0]+(CalibrationData[ClosestIndex+std,0,0]-CalibrationData[ClosestIndex-1+std,0,0])*tu
            DAg[:,std]=np.sum(EuclidDist,0)[:,std]   
        Dstk[0+(std*100):100+(std*100)]=np.sum(EuclidDist,0)[:,std]
        #plot(arange(100)+(100*std),DAg[:,std])
        #pInt=(CalibrationData[ClosestIndex+std,0,0]-CalibrationData[ClosestIndex-1+std,0,0])*np.argmin(DAg[:,std])/100.0+CalibrationData[ClosestIndex-1+std,0,0]
        #print "Best="+str(pInt)+"(d="+str(np.amin(DAg[:,std]))+") from "+str(CalibrationData[ClosestIndex-1+std,0,0])+" to "+str(CalibrationData[ClosestIndex+std,0,0])
    pGlob=(CalibrationData[ClosestIndex+1,0,0]-CalibrationData[ClosestIndex-1,0,0])*np.argmin(Dstk)/200.0+CalibrationData[ClosestIndex-1,0,0]
    pGlobED=np.amin(Dstk)
    #print "Best="+str(pGlob)+"(d="+str(np.amin(Dstk))+") from "+str(CalibrationData[ClosestIndex-1,0,0])+" to "+str(CalibrationData[ClosestIndex+1,0,0])    
    
    # #for RGB color space initial cc is 0
    # refLAB=standardSwatchStats[0:3,0,0,startIndex:endIndex]
    # samLAB=parameterStats[0:3,0,0,0:4]
    # pads=standardSwatchStats[14,0,0,startIndex:endIndex]
    # numPads=samLAB.shape[1]
    # numRefsPads=refLAB.shape[1]
    # numRefs=int(numRefsPads/numPads)
    # distances=np.zeros((numRefs))
    # pHrefs=np.zeros((numRefs))
    # for pad in range(numPads):
    #     cc1=samLAB[:,pad]
    #     padMask=pad==pads
    #     refSamePad=refLAB[:,padMask]
    #     for ref in range(numRefs):
    #         cc2=refSamePad[:,ref]
    #         distances[ref]=distances[ref]+EuclidianDistance(cc1,cc2)
    #         pHrefs[ref]=pHs[padMask][ref]
    # ax.plot(pHrefs,distances)
"""
