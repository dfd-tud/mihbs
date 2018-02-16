#!/usr/bin/env python3

import numpy as np
import cv2


def handle_flip(aInputImage):
    # Note: 255 is white

    # convert to grayscale
    aReferenceImage = cv2.cvtColor(aInputImage, cv2.COLOR_BGR2GRAY)

    # downscale image to 16x16 pixels
    aReferenceImage = cv2.resize(
        aReferenceImage, (16, 16), interpolation=cv2.INTER_AREA)

    oMedianValues = {}

    oMedianValues['tl'] = np.median(
        aReferenceImage[:8, :8])
    oMedianValues['tr'] = np.median(
        aReferenceImage[:8, 8:])
    oMedianValues['bl'] = np.median(
        aReferenceImage[8:, :8])
    oMedianValues['br'] = np.median(
        aReferenceImage[8:, 8:])
    # print("top left: %s" % (str(oMedianValues['tl'])))
    # print("top right: %s" % (str(oMedianValues['tr'])))
    # print("bottom left: %s" % (str(oMedianValues['bl'])))
    # print("buttom right: %s" % (str(oMedianValues['br'])))

    sDarkestCorner = min(oMedianValues, key=lambda i: oMedianValues[i])
    # print("darkest: %s" % str(sDarkestCorner))

    if sDarkestCorner == "tr":
        aInputImage = np.fliplr(aInputImage)
    elif sDarkestCorner == "bl":
        aInputImage = np.flipud(aInputImage)
    elif sDarkestCorner == "br":
        aInputImage = np.fliplr(np.flipud(aInputImage))

    return aInputImage
