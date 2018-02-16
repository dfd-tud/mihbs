#!/usr/bin/env python3

import sys
sys.path.append('../common')

import util
import stability_benchmark as stb


if __name__ == '__main__':
    # create new benchmark instance
    stabbench = stb.StabilityBenchmark()

    # define a set of attacks
    aAttacks = []

    import attacks as at

    # add scale uniform and nonuniform in one direction
    for dScaleValue in [0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2, 4]:
        # define scale uniform and in x and y direction
        aTmpAttack = [(at.scale, {"lScalefactorX": dScaleValue, "lScaleFactorY": dScaleValue}),
                      (at.scale, {"lScalefactorX": dScaleValue,
                                  "lScaleFactorY": 1}),
                      (at.scale, {
                       "lScalefactorX": 1, "lScaleFactorY": dScaleValue})
                      ]
        aAttacks.extend(aTmpAttack)

    # add nonuniform scale
    for dValue1, dValue2 in [(0.25, 0.5), (0.25, 2), (2, 4), (0.9, 1.1)]:
        aTmpAttack = [(at.scale, {"lScalefactorX": dValue1, "lScaleFactorY": dValue2}),
                      (at.scale, {"lScalefactorX": dValue2, "lScaleFactorY": dValue1})]
        aAttacks.extend(aTmpAttack)

    # add rotation attacks
    for lBigAngle in [0, 90, 180, 270, 360]:
        for lSmallAngle in range(-5, 6):
            lAngle = lBigAngle + lSmallAngle
            if not(0 <= lAngle < 360):
                continue
            aTmpAttack = [
                (at.rotation, {"dRotationAngle": lAngle, "bFit": True}),
                (at.rotation, {"dRotationAngle": lAngle, "bFit": False}),
                (at.rotation_cropped, {"dRotationAngle": lAngle})
            ]
            aAttacks.extend(aTmpAttack)

    # add uniform crop
    for lCropPercentage in [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
        tpTmpAttack = (at.crop_percentage, {"tpSlice": (
            lCropPercentage, lCropPercentage, lCropPercentage, lCropPercentage)})
        aAttacks.append(tpTmpAttack)

    # add shift
    for lShiftPixels in [1, 2, 5, 7, 10, 15, 20]:
        aTmpAttack = [
            (at.shift_vertical, {"lPixles": lShiftPixels}),
            (at.shift_horizontal, {"lPixles": lShiftPixels}),
            (at.shift_vertical, {"lPixles": - lShiftPixels}),
            (at.shift_horizontal, {"lPixles": - lShiftPixels}),
        ]
        aAttacks.extend(aTmpAttack)

    # add contrast
    for lContrastValue in range(-70, 71, 5):
        tpTmpAttack = (at.contrast, {"lContrast": lContrastValue})
        aAttacks.append(tpTmpAttack)

    # add gamma adjustment
    for dGamma in [0.25, 0.3, 0.4, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5]:
        tpTmpAttack = (at.gamma_adjustment, {"dGamma": dGamma})
        aAttacks.append(tpTmpAttack)

    # add filter
    for lKernelSize in range(3, 10, 2):
        aTmpAttack = [
            (at.median_filter, {"lKernelSize": lKernelSize}),
            (at.gaussian_filter, {"lSigma": lKernelSize}),
        ]
        aAttacks.extend(aTmpAttack)

    # add jpeg compression
    for lCompressionValue in range(0, 101, 10):
        tpTmpAttack = (at.jpeg_compression, {
                       "lJPEGQuality": lCompressionValue})
        aAttacks.append(tpTmpAttack)

    # add noise
    for dSigma in [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.11, 0.15, 0.2]:
        aTmpAttack = [
            (at.gauss_noise, {"dSigma": dSigma}),
            (at.speckle_noise, {"dSigma": dSigma}),
            (at.salt_and_pepper_noise, {"dAmount": dSigma})
        ]
        aAttacks.extend(aTmpAttack)

    # add brightness
    for lBrightness in range(-120, 121, 10):
        tpTmpAttack = (at.brightness, {"lBrightness": lBrightness})
        aAttacks.append(tpTmpAttack)

    # add handmade attacks
    aHandMadeAttacks = [
        # paper blending
        (at.blend_pattern, {"aPatternImage": util.load_image(
            "../common/whitePaper.tiff")}),
        (at.blend_pattern, {"aPatternImage": util.load_image(
            "../common/recyclePaper.tiff")}),
        # white paper, recycle paper

        # crop percentage - nonuniform
        (at.crop_percentage, {"tpSlice": (0.10, 0, 0, 0)}),
        (at.crop_percentage, {"tpSlice": (0, 0.10, 0, 0)}),
        (at.crop_percentage, {"tpSlice": (0, 0, 0.10, 0)}),
        (at.crop_percentage, {"tpSlice": (0, 0, 0, 0.10)}),
        (at.crop_percentage, {"tpSlice": (0.10, 0.10, 0, 0)}),
        (at.crop_percentage, {"tpSlice": (0, 0.10, 0.10, 0)}),
        (at.crop_percentage, {"tpSlice": (0, 0, 0.10, 0.10)}),
        (at.crop_percentage, {"tpSlice": (0.10, 0, 0, 0.10)}),
        (at.crop_percentage, {
            "tpSlice": (0.1, 0.1, 0.05, 0.05)}),
        (at.crop_percentage, {
            "tpSlice": (0.05, 0.05, 0.1, 0.1)}),

        # flip
        (at.flip, {"bVertical": False}),
        (at.flip, {"bVertical": True}),
    ]

    # add hand defined attacks to attacks list
    aAttacks.extend(aHandMadeAttacks)

    # define a set of hashing algorithms
    # NOTE: add your own hashing algorithms here
    import wuhash as wu
    import blockhash as block
    import predef_hashes as pdh
    import histohash as ht
    import aqcslbp as aq

    aHashes = [
        (pdh.average_hash, {}),
        (pdh.phash, {}),
        (pdh.dhash, {}),
        (pdh.whash, {}),
        (block.blockhash, {}),
        # (wu.wuhash, {}),
        # add rotation handling to wuhash
        (wu.wuhash, {"bRotationHandling": True}),
        # add fliphandling to all hashing methods
        #(pdh.average_hash, {"bFlipHandling": True}),
        # (pdh.phash, {"bFlipHandling": True}),
        #(pdh.dhash, {"bFlipHandling": True}),
        # (pdh.whash, {"bFlipHandling": True}),
        #(block.blockhash, {"bFlipHandling": True}),
        # (wu.wuhash, {"bFlipHandling": True}),
        # (wu.wuhash, {"bFlipHandling": True,
        #              "bRotationHandling": True}),
        (ht.hist_xiang, {}),
        (aq.aq_cslbp, {}),
    ]

    # define the deviation function if whished
    import deviation as dv
    fnDeviationFunction = dv.hamming_distance

    # set number of threads
    lNumberOfThreads = 10

    # set path of image folder
    sPathToImages = "../imagedatasets/stability/"

    # ---- add definitions to benchmark
    stabbench.set_attacks(aAttacks)
    stabbench.set_hashes(aHashes)
    stabbench.set_deviation_fn(fnDeviationFunction)
    stabbench.set_nr_of_threads(lNumberOfThreads)

    # -------- run test -------
    stabbench.run_test_on_images(sPathToImages)
