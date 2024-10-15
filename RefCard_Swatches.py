import numpy as np
import cv2
#import qrcode
#from PIL import Image
font = cv2.FONT_HERSHEY_SIMPLEX

settingString='''
{'ROI xy': array([ 30, 300]), 'CAM exp': array([ 1, 50]), 'WBR xy': array([810, 370]), 'yax ch': array([4]), 'xax sc': array([1, 0, 0]), 'box ul': array([100, 255, 255]), 'WBR sc': array([0]), 'yax sc': array([1, 0, 0]), 'CAM bcs': array([128, 128, 128]), 'hue lo': array([180., 150.]), 'c34 ul': array([ 40, 255, 255]), 'c12 ll': array([140,  20,  40]), 'CAM foc': array([ 1, 60]), 'ROI wh': array([920, 590]), 'xax ch': array([31]), 'c12 ul': array([160, 255, 255]), 'box ll': array([80, 20, 40]), 'ROI ll': array([ 0, 70,  0]), 'c34 ll': array([20, 60, 40]), 'WBR ul': array([255,  30, 255]), 'WBR wh': array([270, 450]), 'ROI ul': array([180, 255, 255]), 'WBR ll': array([0, 0, 0])}
'''

#qrimg = qrcode.make(settingString)
#open_qr_image = np.array(qrimg.convert('RGB'))
#open_qr_image = cv2.resize(open_qr_image, (400, 400))
paperWidth=1800
paperHeight=1200
borderMargin=200
circler=np.uint(borderMargin/3)
circlePad=np.uint(borderMargin/2)
numberSwatches=6
refSwatchSpacing=np.uint((paperHeight-borderMargin)/numberSwatches)
refSwatchDimension=np.uint((paperHeight-borderMargin)/(numberSwatches*1.5))
swatchMargin=np.uint(borderMargin/10)
ReferenceImage = np.full((paperHeight,paperWidth, 3), 255,np.uint8)

#cyan border
cv2.rectangle(ReferenceImage, (0,0), (borderMargin,paperHeight), (255,255,0), -1) #left side of rectangle
cv2.rectangle(ReferenceImage, (0, paperHeight-borderMargin), (paperWidth,paperHeight), (255,255,0), -1) #bottom rectangle in digital view
cv2.rectangle(ReferenceImage, (paperWidth-borderMargin,0), (paperWidth,paperHeight), (255,255,0), -1) #right side of rectangle

#yellowCircles on the left
cv2.circle(ReferenceImage,(circlePad,circlePad), circler, (0,255,255), -1) 
cv2.circle(ReferenceImage,(circlePad,paperHeight-circlePad), circler, (0,255,255), -1) 
#magentaCircles on the right
cv2.circle(ReferenceImage,(paperWidth-circlePad,circlePad), circler, (255,0,255), -1) 
cv2.circle(ReferenceImage,(paperWidth-circlePad,paperHeight-circlePad), circler, (255,0,255), -1) 
circleText="Yellow: ("+str(circlePad)+","+str(circlePad)+")&("+str(circlePad)+","+str(paperHeight-circlePad)+")"
circleText=circleText+" ; Magenta: ("+str(paperWidth-circlePad)+","+str(circlePad)+")&("+str(paperWidth-circlePad)+","+str(paperHeight-circlePad)+")"
cv2.putText(ReferenceImage, circleText, (borderMargin+swatchMargin,paperHeight-borderMargin-swatchMargin), font, 1,(0,0,0),1,cv2.LINE_AA)


colorsRightSwatchs=[[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0],[255,0,255]]

for swatchNum in range(numberSwatches):
    r=(255/numberSwatches*(swatchNum+1))
    g=r
    b=r
    swatchLeftColor=[b,g,r]
    swatchRightColor=colorsRightSwatchs[swatchNum]
    cv2.rectangle(ReferenceImage, (borderMargin+swatchMargin,swatchNum*refSwatchSpacing+swatchMargin), (borderMargin+swatchMargin+refSwatchDimension,swatchNum*refSwatchSpacing+refSwatchDimension+swatchMargin), swatchLeftColor , -1) #left side of rectangle
    cv2.rectangle(ReferenceImage, (paperWidth-borderMargin-swatchMargin-refSwatchDimension,swatchNum*refSwatchSpacing+swatchMargin), (paperWidth-borderMargin-swatchMargin,swatchNum*refSwatchSpacing+refSwatchDimension+swatchMargin), swatchRightColor , -1) #left side of rectangle


#ReferenceImage[395:395+open_qr_image.shape[0],1090:1090+open_qr_image.shape[1],:]=open_qr_image
#cv2.putText(ReferenceImage,"Iodination",(1100,380), font, 1,(0,0,0),1,cv2.LINE_AA)
cv2.imshow('RefCard', ReferenceImage)
keypress=cv2.waitKey(0) & 0xFF
cv2.imwrite("ArchRefCard.jpg", ReferenceImage)
cv2.destroyAllWindows()
