#!/usr/bin/env python3


import numpy as np
import cv2


def hist_xiang(aInputImage):
    '''Histogram-Based Image Hashing Scheme Robust Against Geometric
    Deformations
    Xiang et al. 2007 MM&Sec ACM

    Perceptual Hash with Histogram
    1 Low Pass Filtering (Gauss)
    2 Histogram Extraction
    3 Hash Generation
    '''
    # convert image to grayscale
    aInputImage = cv2.cvtColor(aInputImage, cv2.COLOR_BGR2GRAY)
    # Gaussian Low Pass (k=3, sigma=3)
    aInputImage = cv2.GaussianBlur(aInputImage, (3, 3), 3)

    # Extract Histogram
    len_bins = 30
    avg = np.mean(aInputImage)
    sigma = 0.55
    # padded pixels deducted in paper for rotation
    range_bottom = (1 - sigma) * avg
    range_top = (1 + sigma) * avg
    xiang_range = [range_bottom, range_top]
    histr = cv2.calcHist([aInputImage], [0], None, [len_bins], xiang_range)
    aResult = np.zeros(435, dtype=bool)
    lPosition = 0
    for i in range(0, 29):
        for j in range(i + 1, 30):
            if histr[j][0] == 0:
                a = histr[i][0]
            else:
                a = histr[i][0] / histr[j][0]
            if(a >= 1):
                aResult[lPosition] = True
            lPosition += 1

    return aResult
