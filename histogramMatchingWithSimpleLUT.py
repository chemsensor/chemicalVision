# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:33:44 2024

@author: kevin
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import cv2

def hist_match(source, template):
    # Compute the histograms and their normalized CDFs
    src_hist, _ = np.histogram(source.ravel(), 256, [0,256])
    tgt_hist, _ = np.histogram(template.ravel(), 256, [0,256])
    src_cdf = np.cumsum(src_hist) / float(source.size)
    tgt_cdf = np.cumsum(tgt_hist) / float(template.size)
    
    # Create a mapping from source values to target values
    table = np.interp(src_cdf, tgt_cdf, np.arange(256))
    
    # Apply the mapping to the source image
    return cv2.LUT(source, table.astype(np.uint8))

source = cv2.imread('source.jpg', 0)  # Load in grayscale
reference = cv2.imread('reference.jpg', 0)  # Load in grayscale

matched = hist_match(source, reference)

# Show images
cv2.imshow('Source', source)
cv2.imshow('Reference', reference)
cv2.imshow('Matched', matched)
cv2.waitKey(0)
cv2.destroyAllWindows()