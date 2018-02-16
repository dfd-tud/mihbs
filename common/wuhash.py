#!/usr/bin/env python3

# #######################################################################################################
# ######################################### WuHash ####
# #######################################################################################################
#
# V 0.1
# In case of problems contact: robin.herrmann@tu-dresden.de
#


import numpy as np

from skimage.transform import radon
from scipy.signal import argrelextrema
from fractions import gcd

import cv2
import bisect
import pywt
import fliphandling


def __resize_image(aInputImage, lImageWidth, lImageHeight):
    """resizing an image to a given size"""
    return cv2.resize(aInputImage, (lImageWidth, lImageHeight), interpolation=cv2.INTER_AREA)


def __downscale_image(aInputImage, dScaleFactor=1):
    """resizing an image to a given size"""
    return cv2.resize(aInputImage, None, fx=dScaleFactor, fy=dScaleFactor, interpolation=cv2.INTER_AREA)


def __convert_image_to_grayscale(aInputImage):
    """converting an image to grayscale"""
    return cv2.cvtColor(aInputImage, cv2.COLOR_BGR2GRAY)


def __remove_zero_rows(aSinogram):
    rows_to_delete = []
    rows = aSinogram.shape[0]
    for row in range(rows):
        count = np.count_nonzero(aSinogram[row, :])
        if count == 0:
            rows_to_delete.append(row)
    return np.delete(aSinogram, rows_to_delete, axis=0)


def __radon_rotation(aSinogram, lDegree):
    aSinogram = np.roll(aSinogram, lDegree)
    if(lDegree > 0):
        aSinogram[:, :lDegree] = np.flipud(aSinogram[:, :lDegree])
    else:
        aSinogram[:, lDegree:] = np.flipud(aSinogram[:, lDegree:])
    return aSinogram


def __rotation_detection(aSinogram):
    # Middle Detection
    aDistanceTop = np.zeros(180)
    aDistanceBottom = np.zeros(180)
    for deg in range(180):
        aReferenceLine = aSinogram[:, deg]
        aDistanceTop[deg] = bisect.bisect_left(aReferenceLine, 1)
        aDistanceBottom[deg] = bisect.bisect_left(
            list(reversed(aReferenceLine)), 1)

    # calculate the maximum points and get their position
    aLocalMaximumTop = argrelextrema(aDistanceTop, np.greater)[0]
    aLocalMaximumBottom = argrelextrema(aDistanceBottom, np.greater)[0]

    # filter positions that are wider than a 45 shift in one direction
    aLocalMaximumTop = aLocalMaximumTop[np.logical_and(
        aLocalMaximumTop >= 45, aLocalMaximumTop < 135)]
    aLocalMaximumBottom = aLocalMaximumBottom[np.logical_and(
        aLocalMaximumBottom >= 45, aLocalMaximumBottom < 135)]

    # print("Maxima Top: " + str(aLocalMaximumTop))
    # print("Maxima Bottom: " + str(aLocalMaximumBottom))

    if len(aLocalMaximumBottom) == 1 and len(aLocalMaximumTop) == 1 and (aLocalMaximumBottom[0] == aLocalMaximumTop[0]):
        #print("Yuhaaay, the middle was once " + str(aLocalMaximumTop[0]))
        #print("Angle correction is: " + str(90 - aLocalMaximumTop[0]))
        return 90 - aLocalMaximumTop[0]
    else:
        return 0


def __rotation_handling(aSinogram):
    lCorrectionAngle = __rotation_detection(aSinogram)
    if lCorrectionAngle:
        aSinogram = __radon_rotation(aSinogram, lCorrectionAngle)
    return aSinogram


def __lcm(x, y):
    """This function takes two
    integers and returns the L.C.M."""
    lcm = (x * y) // gcd(x, y)
    return lcm


# this function produces nearly the same results as resize...the only
# differnece is that resize casts to int thus the maximal variation is
# >0.5 per pixeltests
def __calculate_mean_block_image(aInputImage, tpTargetSize=(40, 20)):
    """ calculates the mean blocks as describes in the algo ... but it is
    way slower than the image resize algo...use this instead...the outcome is nearlythe same"""
    lOriginalY = aInputImage.shape[0]
    lOriginalX = aInputImage.shape[1]
    lTargetX, lTargetY = tpTargetSize
    if lTargetX >= lOriginalX or lTargetY >= lOriginalY:
        raise Exception("Can not upscale image, chose a bigger image")

    # handle x axis
    if lOriginalX % lTargetX != 0:
        lLcmX = __lcm(lOriginalX, lTargetX)
        lMultiplierX = int(lLcmX / lOriginalX)
        lBlockSizeX = int(lLcmX / lTargetX)
        aInputImage = np.repeat(aInputImage, lMultiplierX, axis=1)
    else:
        lBlockSizeX = int(lOriginalX / lTargetX)
    lNewY = lOriginalY
    # handle y axis
    if lOriginalY % lTargetY != 0:
        lLcmY = __lcm(lOriginalY, lTargetY)
        lMultiplierY = int(lLcmY / lOriginalY)
        lNewY = int(lOriginalY * lMultiplierY)
        lBlockSizeY = int(lLcmY / lTargetY)
        aInputImage = np.repeat(aInputImage, lMultiplierY, axis=0)
    else:
        lBlockSizeY = int(lOriginalY / lTargetY)

    aBlocks = aInputImage.reshape(lNewY // lBlockSizeY, lBlockSizeY, -1,
                                  lBlockSizeX).swapaxes(1, 2).reshape(-1, lBlockSizeY, lBlockSizeX)
    aMeanBlocks = np.array([np.median(m)
                            for m in aBlocks]).reshape(lTargetY, lTargetX)
    return aMeanBlocks


def wuhash(aInputImage, lMaxSideSize=500, bRotationHandling=False, bFlipHandling=False, bDebug=False):

    # fliphandling
    if bFlipHandling:
        aInputImage = fliphandling.handle_flip(aInputImage)

    if bDebug:
        import matplotlib.pyplot as plt
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)
              ) = plt.subplots(3, 2, figsize=(10, 15))

    # handling of to big images - custom made - not part of the algo
    lImgHeight = aInputImage.shape[0]
    lImgWidth = aInputImage.shape[1]
    if lImgHeight > lMaxSideSize or lImgWidth > lMaxSideSize:
        #print("imput image to big for radon transform, performing downscale")
        dScaleFactor = lMaxSideSize / max(lImgHeight, lImgWidth)
        aInputImage = __downscale_image(aInputImage, dScaleFactor=dScaleFactor)

    aGrayScaleImage = __convert_image_to_grayscale(aInputImage)

    # ----- step 1 :: Radon transform
    aTheta = np.arange(180)
    aSinogram = radon(aGrayScaleImage, theta=aTheta, circle=False)
    aSinogram = __remove_zero_rows(aSinogram)

    if bDebug:
        ax1.set_title("Sinogram")
        ax1.imshow(aSinogram, cmap=plt.cm.Greys_r, aspect='auto')
        ax1.set_xlabel(r"$\theta$")
        ax1.set_ylabel(r"$\rho$")
        ax1.set_xticks(np.arange(0, 181, 15))

    # --- rotation handling
    if bRotationHandling:
        aSinogram = __rotation_handling(aSinogram)

    if bDebug:
        ax2.set_title("Sinogram rotated")
        ax2.imshow(aSinogram, cmap=plt.cm.Greys_r, aspect='auto')
        ax2.set_xlabel(r"$\theta$")
        ax2.set_ylabel(r"$\rho$")
        ax2.set_xticks(np.arange(0, 181, 15))

    # ------ step 2 :: 800 Blocks - mean value

    aMeanBlocks = __resize_image(aSinogram, 20, 40)
    # Note: this is not the original implementation, but it is very close to it

    if bDebug:
        ax3.set_title("Mean Blocks")
        ax3.imshow(aMeanBlocks, cmap=plt.cm.Greys_r, aspect='auto')

    # ---------- step 3 :: 2 level haar wavelet transform
    aWaveletHeightFrequencies = np.zeros((20, 20))

    for nr_col in range(aMeanBlocks.shape[1]):
        # take high frequency part only
        aWaveletHeightFrequencies[:, nr_col] = pywt.wavedec(
            aMeanBlocks[:, nr_col], "Haar", level=2)[2]

    if bDebug:
        ax4.set_title("Wavelet\n(high frequency components)")
        ax4.imshow(aWaveletHeightFrequencies,
                   cmap=plt.cm.Greys_r, aspect='auto')

    # ----- step 4 fft reals part
    aFFT = np.zeros(aWaveletHeightFrequencies.shape)
    for nr_col in range(aWaveletHeightFrequencies.shape[1]):
        aFFT[:, nr_col] = np.real(np.fft.fft(
            aWaveletHeightFrequencies[:, nr_col]))

    if bDebug:
        ax5.set_title("FFT\n(reals components)")
        ax5.imshow(aFFT, cmap=plt.cm.Greys_r, aspect='auto')

    # ----- step 6 - calculate mean and hash

    lFFTHeight, lFFTWidth = aFFT.shape
    aHashBlock = np.zeros(aFFT.shape, dtype=bool)

    #dMeanThreshold = np.mean(aFFT)
    for nr_col in range(lFFTWidth):
        dMeanThreshold = np.mean(aFFT[:, nr_col])
        for nr_row in range(lFFTHeight):
            if aFFT[nr_row, nr_col] >= dMeanThreshold:
                aHashBlock[nr_row, nr_col] = 1

    if bDebug:
        ax6.set_title("Hash")
        ax6.imshow(aHashBlock, cmap=plt.cm.Greys_r, aspect='auto')

    if bDebug:
        fig.tight_layout()
        plt.show()
        plt.savefig("test_wu_hash.png")

    # return the hash flattened columnwise
    return aHashBlock.flatten('F')
