#!/usr/bin/env python3

import numpy as np

import cv2
# from scipy import ndimage
from skimage import exposure
from blend_modes import blend_modes
from skimage import util
from skimage import img_as_float
from skimage import img_as_ubyte
import math


def scale(aInputImage, lScalefactorX=1, lScaleFactorY=1):
    """scaling an image by a given scale factor"""
    if (lScalefactorX <= 1 and lScaleFactorY <= 1):
        fnInterpolation = cv2.INTER_AREA
    else:
        fnInterpolation = cv2.INTER_CUBIC
    return cv2.resize(aInputImage, None, fx=lScalefactorX, fy=lScaleFactorY, interpolation=fnInterpolation)


def rotation_cropped(aInputImage, dRotationAngle=25):
    """ perform a cropped rotation that crops the rotated image to fill the whole canvas without borders"""
    def rotatedRectWithMaxArea(w, h, angle):
        # thanks to coproc
        # (https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders)
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
            return 0, 0

        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of
        # sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer
            # line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x /
                      cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / \
                cos_2a, (h * cos_a - w * sin_a) / cos_2a

        return int(wr), int(hr)

    lOriginalY = aInputImage.shape[0]
    lOriginalX = aInputImage.shape[1]
    # rotate about the middle point
    M = cv2.getRotationMatrix2D(
        center=((lOriginalX - 1) / 2, (lOriginalY - 1) / 2), angle=dRotationAngle, scale=1.0)
    lNewX, lNewY = rotatedRectWithMaxArea(
        lOriginalX, lOriginalY, math.radians(dRotationAngle))
    (tx, ty) = ((lNewX - lOriginalX) / 2, (lNewY - lOriginalY) / 2)
    # third column of matrix holds translation, which takes effect after
    # rotation.
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(aInputImage, M, dsize=(lNewX, lNewY))


def rotation(aInputImage,  dRotationAngle=30, bFit=True, tpBorderValue=(0, 0, 0)):
    # thanks to Lars Schillingmann (https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c)
    # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    lOriginalY = aInputImage.shape[0]
    lOriginalX = aInputImage.shape[1]
    # rotate about the middle point
    M = cv2.getRotationMatrix2D(
        center=((lOriginalX - 1) / 2, (lOriginalY - 1) / 2), angle=dRotationAngle, scale=1.0)

    if bFit:
        # calculate new image size
        lNewX, lNewY = lOriginalX, lOriginalY
        # include this if you want to prevent corners being cut off
        dAngle = np.deg2rad(dRotationAngle)
        lNewX, lNewY = (abs(np.sin(dAngle) * lNewY) + abs(np.cos(dAngle) * lNewX),
                        abs(np.sin(dAngle) * lNewX) + abs(np.cos(dAngle) * lNewY))

        (tx, ty) = ((lNewX - lOriginalX) / 2, (lNewY - lOriginalY) / 2)
        # third column of matrix holds translation, which takes effect after
        # rotation.
        M[0, 2] += tx
        M[1, 2] += ty

        aRotatedImg = cv2.warpAffine(
            aInputImage, M, dsize=(int(lNewX), int(lNewY)), borderValue=tpBorderValue)
    else:
        aRotatedImg = cv2.warpAffine(
            aInputImage, M, dsize=(lOriginalX, lOriginalY), borderValue=tpBorderValue)
    return aRotatedImg


def flip(aInputImage, bVertical=False):
    if(bVertical):
        return np.flipud(aInputImage)
    else:
        return np.fliplr(aInputImage)


def perspective_transformation(aInputImage, tl=(0, 0), tr=(0, 0), bl=(0, 0), br=(0, 0), bResize=True, tpBorderValue=(0, 0, 0)):
    """transform an image, move the four edge point, for every point you can give (x,y) where x and y can be positive or negative"""

    lImgHeight = aInputImage.shape[0]
    lImgWidth = aInputImage.shape[1]
    rect = np.array([
        [0, 0],
        [lImgWidth - 1, 0],
        [lImgWidth - 1, lImgHeight - 1],
        [0, lImgHeight - 1]
    ], dtype="float32")

    dst = np.array([
        [0 + tl[0], 0 + tl[1]],
        [lImgWidth - 1 + tr[0], 0 + tr[1]],
        [lImgWidth - 1 + br[0], lImgHeight - 1 + br[1]],
        [0 + bl[0], lImgHeight - 1 + bl[1]]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)

    lDeltaWidth = 0
    lDeltaHeight = 0
    if bResize:
        lXTranslation = -min(tl[0], bl[0])
        lYTranslation = -min(tl[1], tr[1])
        lDeltaWidth = lXTranslation + max(br[0], tr[0])
        lDeltaHeight = lYTranslation + max(bl[1], br[1])
        aTranslation = np.array([
            [1, 0, lXTranslation],
            [0, 1, lYTranslation],
            [0, 0, 1]
        ], dtype="float32")
        M = np.dot(aTranslation, M)

    return cv2.warpPerspective(aInputImage, M, (lImgWidth + lDeltaWidth, lImgHeight + lDeltaHeight), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, borderValue=tpBorderValue)


def crop(aInputImage, tpSlice):
    """crops an image, x pixel according to a slice tuple (top, left, buttom, right)"""
    t, l, b, r = tpSlice
    lImgHeight = aInputImage.shape[0]
    lImgWidth = aInputImage.shape[1]
    if t + b >= lImgHeight or l + r >= lImgWidth:
        raise Exception("Crop area can not be bigger that the image.")
    return aInputImage[t:lImgHeight - b, l:lImgWidth - r]


def crop_percentage(aInputImage, tpSlice):
    """crops an image, x percent according to a slice tuple (top, left, buttom, right)"""
    tp, lp, bp, rp = tpSlice
    if tp + bp >= 1 or lp + rp >= 1:
        raise Exception("Crop area can not be bigger that the image.")
    lImgHeight = aInputImage.shape[0]
    lImgWidth = aInputImage.shape[1]
    t = int(np.floor(tp * lImgHeight))
    b = int(np.floor(bp * lImgHeight))
    l = int(np.floor(lp * lImgWidth))
    r = int(np.floor(rp * lImgWidth))

    return crop(aInputImage, (t, l, b, r))

# def colored_noise(aInputImage, mean=5, stddev=128):
# return cv2.randn(np.copy(aInputImage), (mean, mean, mean), (stddev,
# stddev, stddev)) + aInputImage


# def noise_gauss(aInputImage, dSigma=0.1):
#     """ applies gaussian noise to an image """
#     row, col, ch = aInputImage.shape
#     lMean = 0
#     aGauss = np.random.normal(lMean, dSigma, (row, col, ch))
#     aGauss = aGauss.reshape(row, col, ch)
#     return (aInputImage + aGauss).astype(np.dtype(np.uint8))


def brightness(aInputImage, lBrightness=150):
    aBrightedImage = aInputImage.astype(np.int16) + lBrightness
    aBrightedImage[aBrightedImage > 255] = 255
    aBrightedImage[aBrightedImage < 0] = 0
    return aBrightedImage.astype(np.uint8)


def contrast(aInputImage, lContrast=50):
    """ changes the contrast of a given image by histogram manipulation
        takes values between [-128, 128] as contrast factor
    """
    if abs(lContrast) > 128:
        raise Exception("contrast value have to be in range [-128, 128]")
    if lContrast >= 0:
        return exposure.rescale_intensity(aInputImage, in_range=(0 + lContrast, 255 - lContrast))
    else:
        return exposure.rescale_intensity(aInputImage, out_range=(0 + abs(lContrast), 255 - abs(lContrast)))


# def gaussian_filter(aInputImage, dSigma=2.0):
#     """ applies a gaussian filter on a given image """
#     return ndimage.gaussian_filter(aInputImage, dSigma)


# def median_filter(aInputImage, dSigma=3.0):
#     return ndimage.median_filter(aInputImage, dSigma)


def gaussian_filter(aInputImage, lSigma=5):
    """apply a gaussian filter on a given image
       the size of the kernel ist calculated from sigma
    """
    return cv2.GaussianBlur(aInputImage, (0, 0), lSigma)


def median_filter(aInputImage, lKernelSize=5):
    """apply a median filter on a given image"""
    return cv2.medianBlur(aInputImage, lKernelSize)


def add_pattern(aInputImage, aPatternImage, dAlphaPattern=0.2):
    return cv2.addWeighted(aInputImage, 1.0 - dAlphaPattern, aPatternImage, dAlphaPattern, 0.0)


def blend_image(aInputImage, aBGImage, dOpacity=1.0, sBlendMode="mul"):
    """ blending pattern to image"""
    if aInputImage.shape != aBGImage.shape:
        raise Exception(
            "Image and BackgrundImage have to be the same Size and color Model")
    aInputImageAlpha = __add_alpha_channel(aInputImage)
    aBGImageAlpha = __add_alpha_channel(aBGImage)

    dicBlendModes = {
        "sl": blend_modes.soft_light,
        "lo": blend_modes.lighten_only,
        "do": blend_modes.dodge,
        "ad": blend_modes.addition,
        "dar": blend_modes.darken_only,
        "mul": blend_modes.multiply,
        "hl": blend_modes.hard_light,
        "dif": blend_modes.difference,
        "sub": blend_modes.subtract,
        "gre": blend_modes.grain_extract,
        "grm": blend_modes.grain_merge,
        "div": blend_modes.divide}
    if sBlendMode in dicBlendModes.keys():
        fnBlendMode = dicBlendModes[sBlendMode]
    else:
        fnBlendMode = dicBlendModes["mul"]

    aBlendedImageAlpha = fnBlendMode(
        aBGImageAlpha.astype(float), aInputImageAlpha.astype(float), dOpacity)
    aBlendedImage = __remove_alpha_channel(aBlendedImageAlpha)
    return np.uint8(aBlendedImage)


def blend_pattern(aInputImage, aPatternImage=None, dOpacity=1.0, sBlendMode="mul"):
    """ stretches the pattern image to the size of the image and blends them """
    if aPatternImage is None:
        raise Exception("no image was given as pattern")
    aPatternRightSize = __resize_pattern(aInputImage, aPatternImage)
    return blend_image(aInputImage, aPatternRightSize, dOpacity, sBlendMode)


def __resize_pattern(aInputImage, aPatternImage):
    """ resize a pattern image to the whished size"""
    lTargetHeight, lTargetWidth = aInputImage.shape[:2]
    lPatternHeigh, lPatternWidth = aPatternImage.shape[:2]
    # if pattern is already bigger than necessary
    if lTargetHeight < lPatternHeigh and lTargetWidth < lPatternWidth:
        return aPatternImage[:lTargetHeight, :lTargetWidth]
    # calculate multiplier
    lPatternHeighMultiplier = math.ceil(lTargetHeight / lPatternHeigh)
    lPatternWidthMultiplier = math.ceil(lTargetWidth / lPatternWidth)
    lPatternColorMultiplier = 0 if len(aPatternImage.shape) != 3 else 1
    # multiply the pattern
    aMultipliedPattern = np.tile(
        aPatternImage, (lPatternHeighMultiplier, lPatternWidth, lPatternColorMultiplier))
    return aMultipliedPattern[:lTargetHeight, :lTargetWidth]


def __add_alpha_channel(aInputImage):
    """ adds an alpha chanel to the image """
    aBChannel, aGChannel, aRChannel = cv2.split(aInputImage)
    aAlphaChannel = np.ones(aBChannel.shape, dtype=aBChannel.dtype)
    return cv2.merge((aBChannel, aGChannel, aRChannel, aAlphaChannel * 255))


def __remove_alpha_channel(aInputImage):
    """ removes the alpha channel from image """
    return aInputImage[:, :, :3]


def gamma_adjustment(aInputImage, dGamma=1.0, dGain=1.0):
    """ adapts the gamma exposure """
    return exposure.adjust_gamma(aInputImage, dGamma, dGain)


def shift_vertical(aInputImage, lPixles=10, bFillArea=False, tpBorderValue=(0, 0, 0)):
    """ shifts an image by x pixles vertically """
    aShiftedImage = np.roll(aInputImage, lPixles, 0)
    if bFillArea:
        if lPixles >= 0:
            aShiftedImage[:lPixles, :] = tpBorderValue
        else:
            aShiftedImage[lPixles:, :] = tpBorderValue
    return aShiftedImage


def shift_horizontal(aInputImage, lPixles=10, bFillArea=False, tpBorderValue=(0, 0, 0)):
    """ shifts an image by x pixles horizontally """
    aShiftedImage = np.roll(aInputImage, lPixles, 1)
    if bFillArea:
        if lPixles >= 0:
            aShiftedImage[:, :lPixles] = tpBorderValue
        else:
            aShiftedImage[:, lPixles:] = tpBorderValue
    return aShiftedImage


def jpeg_compression(aInputImage, lJPEGQuality=95):
    """ encodes an image as jpeg with a given quality """
    r, aBuffer = cv2.imencode(".jpeg", aInputImage,
                              (cv2.IMWRITE_JPEG_QUALITY, lJPEGQuality))
    return cv2.imdecode(aBuffer, -1)


def speckle_noise(aInputImage, dSigma=0.001):
    """ adds speckle noise """
    image = img_as_float(aInputImage)
    image = util.random_noise(aInputImage, 'speckle', mean=0, var=dSigma)
    return img_as_ubyte(image)


def salt_and_pepper_noise(aInputImage, dAmount=0.001, dProportion=0.5):
    """ adds salt and pepper noise """
    image = img_as_float(aInputImage)
    image = util.random_noise(
        aInputImage, 's&p', amount=dAmount, salt_vs_pepper=dProportion)
    return img_as_ubyte(image)


def gauss_noise(aInputImage, dSigma=0.1):
    """adds gaussian noise with skimage """
    image = img_as_float(aInputImage)
    image = util.random_noise(image, 'gaussian', mean=0, var=dSigma)
    return img_as_ubyte(image)


def position_watermarking(aInputImage, aWatermarkImage, position='top_left', dOpacity=1.0, sBlendMode="mul"):
    """adds Watermark on 1 of 5 positions: top_left, bottom_left, bottom_right, top_right, middle"""
    lTargetHeight, lTargetWidth = aInputImage.shape[0], aInputImage.shape[1]
    aBlankImage = np.zeros((lTargetHeight, lTargetWidth, 3), np.uint8)
    aBlankImage[:, :] = (255, 255, 255)
    lWatermarkHeigh, lWatermarkWidth = aWatermarkImage.shape[:2]
    if position == 'top_left':
        aBlankImage[0:lWatermarkHeigh, 0:lWatermarkWidth] = aWatermarkImage
    elif position == 'bottom_left':
        aBlankImage[(lTargetHeight - lWatermarkHeigh):lTargetHeight,
                    0:lWatermarkWidth] = aWatermarkImage
    elif position == 'bottom_right':
        aBlankImage[(lTargetHeight - lWatermarkHeigh):lTargetHeight,
                    (lTargetWidth - lWatermarkWidth):lTargetWidth] = aWatermarkImage
    elif position == 'top_right':
        aBlankImage[0:lWatermarkHeigh,
                    (lTargetWidth - lWatermarkWidth):lTargetWidth] = aWatermarkImage
    elif position == 'middle':
        if lTargetHeight % 2 == 0 and lTargetWidth % 2 == 0:
            aBlankImage = cv2.resize(
                aBlankImage, (lTargetHeight + 1, lTargetWidth + 1))
        elif lTargetHeight % 2 == 0:
            aBlankImage = cv2.resize(
                aBlankImage, (lTargetHeight + 1, lTargetWidth))
        elif lTargetWidth % 2 == 0:
            aBlankImage = cv2.resize(
                aBlankImage, (lTargetHeight, lTargetWidth + 1))
        if lWatermarkHeigh % 2 == 0 and lWatermarkWidth % 2 == 0:
            aWatermarkImage = cv2.resize(
                aWatermarkImage, (lWatermarkHeigh + 1, lWatermarkWidth + 1))
        elif lWatermarkHeigh % 2 == 0:
            aWatermarkImage = cv2.resize(
                aWatermarkImage, (lWatermarkHeigh + 1, lWatermarkWidth))
        elif lWatermarkWidth % 2 == 0:
            aWatermarkImage = cv2.resize(
                aWatermarkImage, (lWatermarkHeigh, lWatermarkWidth + 1))
        (lCenterImageX, lCenterImageY) = (
            (aBlankImage.shape[0] + 1) / 2, (aBlankImage.shape[1] + 1) / 2)
        (lCenterWatermarkX, lCenterWatermarkY) = (
            (aWatermarkImage.shape[0] + 1) / 2, (aWatermarkImage.shape[1] + 1) / 2)
        X1 = int(lCenterImageX - (lCenterWatermarkX))
        Y1 = int(lCenterImageY - (lCenterWatermarkY))
        aBlankImage[X1:(X1 + aWatermarkImage.shape[0]),
                    Y1:(Y1 + aWatermarkImage.shape[1])] = aWatermarkImage
        aBlankImage = cv2.resize(aBlankImage, (lTargetWidth, lTargetHeight))
    return blend_image(aInputImage, aBlankImage, dOpacity, sBlendMode)
