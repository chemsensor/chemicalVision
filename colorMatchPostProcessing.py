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
    
        
pads=standardSwatchStats[14,0,0,:]
pHs=standardSwatchStats[13,0,0,:]
padList=set(pads)
fig,axes=plt.subplots(3,3,sharex=True,sharey=True)
for padNumber in padList:
    padMask=padNumber==standardSwatchStats[14,0,0,:]
    for cc,col in zip([0,3,6],[0,1,2]):
        axes[int(padNumber),col].plot(pHs[padMask],standardSwatchStats[cc,0,0,:][padMask],'-or')
        axes[int(padNumber),col].plot(pHs[padMask],standardSwatchStats[cc+1,0,0,:][padMask],'-og')
        axes[int(padNumber),col].plot(pHs[padMask],standardSwatchStats[cc+2,0,0,:][padMask],'-ob')

#for LAB color space initial cc is 6
refLAB=standardSwatchStats[6:9,0,0,:]
samLAB=parameterStats[6:9,0,0,0:3]
pads=standardSwatchStats[14,0,0,:]
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

#for RGB color space initial cc is 0
refLAB=standardSwatchStats[0:3,0,0,:]
samLAB=parameterStats[0:3,0,0,0:3]
pads=standardSwatchStats[14,0,0,:]
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
ax.plot(pHrefs,distances)

#for HSV color space initial cc is 3
refLAB=standardSwatchStats[3:6,0,0,:]
samLAB=parameterStats[3:6,0,0,0:3]
pads=standardSwatchStats[14,0,0,:]
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
ax.plot(pHrefs,distances)