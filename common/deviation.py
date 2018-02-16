#!/usr/bin/env python3

# #######################################################################################################
# ######################################### Analysis ####
# #######################################################################################################
#
# V 0.1
# In case of problems contact: robin.herrmann@tu-dresden.de
#
#

import numpy as np


def hamming_distance(array1, array2):
    if (array1.size != array2.size):
        raise Exception("Arrays have to have the same size")
    hamming = array1 != array2
    return np.count_nonzero(hamming) / hamming.size


def segment_distance(aSegmentHashes1, aSegmentHashes2, dThreshold=0.1):
    """calculates the distance between two segmented images
       given their segment hashes and a minimal threshold for the
       hamming distances between two segment to be considered as equal
    """
    aSegmentHashes1 = np.array(aSegmentHashes1)
    aSegmentHashes2 = np.array(aSegmentHashes2)
    aSegmentDeviations = []
    for aSegmentHash1 in aSegmentHashes1:
        for aSegmentHash2 in aSegmentHashes2:
            dSegmentsDeviation = hamming_distance(aSegmentHash1, aSegmentHash2)
            if(dSegmentsDeviation <= dThreshold):
                aSegmentDeviations.append(dSegmentsDeviation)

    # calculate segment distance according to the proposed formula
    return np.mean(aSegmentDeviations) / (len(aSegmentDeviations) * dThreshold * 2) if aSegmentDeviations else 1.0


def segment_distance_precalculated_deviations(aSegmentDeviations, dThreshold=0.1):
    """calculates the distance between two segmented images
       given all segment deviations
    """
    aSegmentDeviations = np.array(aSegmentDeviations)
    aSegmentDeviations = np.extract(
        aSegmentDeviations <= dThreshold, aSegmentDeviations)

    # calculate segment distance according to the proposed formula
    return np.mean(aSegmentDeviations) / (len(aSegmentDeviations) * dThreshold * 2) if aSegmentDeviations.size > 0 else 1.0
