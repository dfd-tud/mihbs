import numpy as np
import cv2


def __convert_to_grayscale(aInputImage):
    dChanNumber = aInputImage.shape
    if len(dChanNumber) > 2:
        return cv2.cvtColor(aInputImage, cv2.COLOR_BGR2GRAY)
    else:
        return aInputImage


def __resize_image(aInputImage, lOutputSize):
    return cv2.resize(aInputImage, lOutputSize)


def __calculate_thresholds(aInput, bDebug=False):
    """ calculate thresholds for DBQ """
    dMean = np.mean(aInput)
    if bDebug:
        print(dMean)
    lS1 = sorted(i - dMean for i in aInput if i <= dMean)
    lS2 = []
    lS3 = sorted(i - dMean for i in aInput if i > dMean)

    dMin = 0

    while lS1 and lS3:
        dSum2 = sum(lS2)
        if dSum2 <= 0:
            dMin3 = min(lS3)
            lS3.remove(dMin3)
            lS2.append(dMin3)
        else:
            dMax1 = max(lS1)
            lS1.remove(dMax1)
            lS2.append(dMax1)

        if lS1 and lS3:
            F = (sum(lS1) ** 2 / len(lS1)
                 + sum(lS3) ** 2 / len(lS3))
        elif lS1 and not lS3:
            F = (sum(lS1) ** 2 / len(lS1))
        elif not lS1 and lS3:
            F = (sum(lS3) ** 2 / len(lS3))

        E = (sum(map(lambda x: (x - np.mean(lS1)) ** 2, lS1))
             + sum(map(lambda x: (x - np.mean(lS2)) ** 2, lS2))
             + sum(map(lambda x: (x - np.mean(lS3)) ** 2, lS3)))

        if F > dMin:
            dMin = F
            if lS1:
                a = max(lS1)
            else:
                a = min(lS2)
            b = max(lS2)
            if bDebug:
                print("save new a, b: ", a + dMean, b + dMean, E, F)

    a = a + dMean
    b = b + dMean

    return a, b


def __calculate_all_possible_E(aInput):
    """ debug function for threshold-estimation,
    calculates values for all possible thresholds"""

    a = 0
    b = len(aInput)
    dMin = float("inf")

    for x in range(len(aInput)):
        for y in range(len(aInput)):
            left = aInput[x]
            right = aInput[y]
            if left < right:
                lS1 = sorted(i for i in aInput if i <= left)
                lS2 = sorted(i for i in aInput if left < i <= right)
                lS3 = sorted(i for i in aInput if i > right)

                E = (sum(map(lambda x: (x - np.mean(lS1)) ** 2, lS1))
                     + sum(map(lambda x: (x - np.mean(lS2)) ** 2, lS2))
                     + sum(map(lambda x: (x - np.mean(lS3)) ** 2, lS3)))

                if E < dMin:
                    dMin = E
                    a = left
                    b = right
                    print("new min: ", a, b, E)

    return a, b


def __calculate_block_coordinates(iRowNumber, iColumnNumber, iBlockSize):
    return (iRowNumber * iBlockSize, (iRowNumber + 1) * iBlockSize,
            iColumnNumber * iBlockSize, (iColumnNumber + 1) * iBlockSize)


def __calculate_abolsute_difference(a, b):
    return abs(int(a) - int(b))


def adaptive_wiener_filter(aImage, neighbors=(2, 2), bDebug=False):
    aImage = __convert_to_grayscale(aImage)
    rows, cols = aImage.shape
    dSumGlobal = 0
    aOutput = np.zeros(shape=(rows, cols))

    # calculate global var (var of noise)
    for row in range(0, rows):
        for col in range(0, cols):
            r0 = row - neighbors[0]
            if r0 < 0:
                r0 = 0
            c0 = col - neighbors[1]
            if c0 < 0:
                c0 = 0

            aBlock = aImage[r0:row + neighbors[0] + 1,
                            c0:col + neighbors[1] + 1]

            dVar = np.var(aBlock)
            dSumGlobal += dVar

    count = rows * cols
    dVarGlobal = dSumGlobal / count

    # calculate filtered image
    for row in range(0, rows):
        for col in range(0, cols):
            # extract neighborhood
            r0 = row - neighbors[0]
            if r0 < 0:
                r0 = 0
            c0 = col - neighbors[1]
            if c0 < 0:
                c0 = 0

            aBlock = aImage[r0:row + neighbors[0] + 1,
                            c0:col + neighbors[1] + 1]

            # calculate mean and var for neighborhood
            dMean = np.mean(aBlock)
            dVar = np.var(aBlock)

            # calculate var_o
            dVarO = max(0, dVar - dVarGlobal)

            # calculate new value
            dAdaptive = int((dMean
                             + (dVarO) / (dVarO + dVarGlobal)
                             * (aImage[row, col] - dMean)))

            aOutput[row, col] = dAdaptive

            if bDebug:
                print(aBlock)
                print("value:  ", aImage[row, col])
                print("mean:   ", dMean)
                print("var:    ", dVar)
                print("dVarO:  ", dVarO)
                print("var-glob:",  dVarGlobal)
                print("fact:   ", (dVarO) / (dVarO + dVarGlobal))
                print("val-mean", aImage[row, col] - dMean)
                print("new:    ", dAdaptive)

    return aOutput.astype(np.uint8)


def double_bit_quantization(lInput, bDebug=False):
    dThreshold1, dThreshold2 = __calculate_thresholds(lInput, bDebug=bDebug)
    if bDebug:
        print(__calculate_all_possible_E(lInput))
    lOut = []
    for i in range(len(lInput)):
        if lInput[i] <= dThreshold1:
            lOut.append([False, True])
        elif lInput[i] <= dThreshold2:
            lOut.append([False, False])
        else:
            lOut.append([True, False])

    flattenedList = [item for sublist in lOut for item in sublist]
    #stringList = map(str, map(int, flattenedList))
    #sOut = ''.join(stringList)

    # return sOut
    return flattenedList


def cslbp(aBlock, T=0.01, bNorm=False, bDebug=False):
    lHist1 = np.zeros(shape=(1, 8))

    for i in range(1, aBlock.shape[0] - 1, 1):
        for j in range(1, aBlock.shape[1] - 1, 1):
            if bDebug:
                print("subblock: ", aBlock[i - 1:i + 2, j - 1:j + 2])
            # calculate sign
            v0 = int(aBlock[i, j + 1]) - int(aBlock[i, j - 1]) > T
            v1 = int(aBlock[i + 1, j + 1]) - int(aBlock[i - 1, j - 1]) > T
            v2 = int(aBlock[i + 1, j]) - int(aBlock[i - 1, j]) > T
            v3 = int(aBlock[i + 1, j - 1]) - int(aBlock[i - 1, j + 1]) > T

            dValue = v0 + v1 * 2 + v2 * 4 + v3 * 8

            # calculate magnitudes
            m04 = __calculate_abolsute_difference(aBlock[i, j + 1],
                                                  aBlock[i, j - 1])
            m15 = __calculate_abolsute_difference(aBlock[i + 1, j + 1],
                                                  aBlock[i - 1, j - 1])
            m26 = __calculate_abolsute_difference(aBlock[i + 1, j],
                                                  aBlock[i - 1, j])
            m37 = __calculate_abolsute_difference(aBlock[i + 1, j - 1],
                                                  aBlock[i - 1, j + 1])

            # calculate Average of magnitudes
            dMeanMag = np.mean([m04, m15, m26, m37])

            if bDebug:
                print(dValue, dMeanMag)

            # fill 8-bin histogram with flip
            if dValue > 7:
                dValue = 15 - dValue
            lHist1[0, dValue] += dMeanMag

    # normalize histogram (not needed with quantization, but mabye faster)
    if bNorm:
        if sum(lHist1[0]) != 0:
            lHist1[0] = lHist1[0] / sum(lHist1[0])

    if bDebug:
        print(lHist1[0])

    # return-value is 1-dim array
    return lHist1[0]


def aq_cslbp(aImage, lScaleTo=(256, 256), dBlockSize=32, T=0.01,
             bNorm=True, bDebug=False):

    # preprocessing (Grayscale, Scaling)
    aImage = __convert_to_grayscale(aImage)
    aImage = __resize_image(aImage, lScaleTo)

    iNumberOfColumns = int(lScaleTo[0] / dBlockSize)
    iNumberOfRows = int(lScaleTo[1] / dBlockSize)

    lHash = []

    # extract imageblocks
    for i in range(iNumberOfRows):
        for j in range(iNumberOfColumns):
            (r0, r1, c0, c1) = __calculate_block_coordinates(i, j, dBlockSize)
            aBlock = aImage[r0:r1, c0:c1]

            # apply adaptive wiener filter
            aBlock = adaptive_wiener_filter(aBlock, bDebug=bDebug)
            # calculate aq-cslbp
            lHash.append(cslbp(aBlock, T=T, bNorm=bNorm, bDebug=bDebug))

    lHash = [item for sublist in lHash for item in sublist]

    # quantize with DBQ
    lHashString = np.array(double_bit_quantization(lHash, bDebug=bDebug))

    return lHashString
