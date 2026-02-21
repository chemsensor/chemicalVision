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
    
    # closestIndex=np.argmin(distances)
    # closest_pH=pHrefs[closestIndex]
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