#!/usr/bin/env python3

# #######################################################################################################
# ######################################### Blockhash ####
# #######################################################################################################
#
# V 0.1
# In case of problems contact: robin.herrmann@tu-dresden.de
#
#

import math
import numpy as np
import fliphandling

import cv2


def __resize_image(aInputImage, lImageWidth, lImageHeight):
    """resizing an image to a given size"""
    return cv2.resize(aInputImage, (lImageWidth, lImageHeight), interpolation=cv2.INTER_AREA)


def __convert_image_to_grayscale(aInputImage):
    """converting an image to grayscale"""
    return cv2.cvtColor(aInputImage, cv2.COLOR_BGR2GRAY)


def __is_odd(lNumber):
    return lNumber & 0x1


def __show_image(aOutputImage):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", aOutputImage)
    cv2.waitKey()
    cv2.destroyAllWindows()


def blockhash(aInputImage, lHashSize=256, bFlipHandling=False, bDebug=False):
    dImgDimension = math.sqrt(lHashSize)
    if not float(dImgDimension).is_integer():
        raise Exception(
            "The squareroot of the hash size has to be an int")
    lImgDimension = int(dImgDimension)
    if __is_odd(lImgDimension):
        raise Exception("The squareroot of the hash size has to be even")

    # fliphandling
    if bFlipHandling:
        aInputImage = fliphandling.handle_flip(aInputImage)

    if bDebug:
        __show_image(aInputImage)
    aGrayScaleImage = __convert_image_to_grayscale(aInputImage)
    aWorkingImage = __resize_image(
        aGrayScaleImage, lImgDimension, lImgDimension)

    lHeight, lWidth = aWorkingImage.shape
    # you can use it here if your blocks are single pixels only
    dMedianValue = np.median(aWorkingImage)

    aBlockHash = np.zeros((lImgDimension, lImgDimension), dtype=bool)

    for x in range(lHeight):
        for y in range(lWidth):
            # if block is bigger than one pixel, use mean of block
            if aWorkingImage[x, y] >= dMedianValue:
                aBlockHash[x, y] = True

    return aBlockHash.flatten()

    # TODO: return an array of hashes if the values of the flip handling are
    # to equal

    # NOTE: if using bigger blocks
    # thresh = np.zeros((512,512), dtype=np.uint8)
    # klaus_bw [:2,:2]
    # np.median(klaus_bw [:2,:2])
