# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:46:11 2020

@author: cantr
"""


def onmouse(event,x,y,flags,params):
    global ColorRectangle,ColorRect,ix,iy,ixg,iyg,ColorRectOver,GreyRectangle,GreyRect,GreyRectOver,rebalanceToggle
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
            cv2.waitKey(1)
        if GreyRectangle == True:
            cv2.rectangle(img,(ixg,iyg),(x,y),(0,0,0),2)
            GreyRect = (min(ixg,x),min(iyg,y),abs(ixg-x),abs(iyg-y))
            cv2.imshow('Result',img)
            cv2.waitKey(1)
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


cv2.namedWindow('Result',cv2.WINDOW_GUI_NORMAL) 
cv2.setMouseCallback('Result',onmouse) 