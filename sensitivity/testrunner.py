#!/usr/bin/env python3

import sys
sys.path.append('../common')

import sensitivity_benchmark as seb


if __name__ == '__main__':
    # create new benchmark instance
    sensbench = seb.SensitivityBenchmark()

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

    # set number of threads
    lNumberOfThreads = 50

    # set pathes to imagesets that should be hashed
    aImagesets = [
        "../imagedatasets/sensitivity/dogs/",
        "../imagedatasets/sensitivity/flowers/"
    ]

    # ---- add definitions to benchmark
    sensbench.set_hashes(aHashes)
    sensbench.set_nr_of_threads(lNumberOfThreads)

    # -------- run test -------
    # NOTE: If you already hashed an imageset and you want to test an additional
    # hashing algorithm without hashing the dataset again with the previous defined hashing
    # algorithms, than set the bAddSave=True. In this mode the algorithm will check for every
    # single image whether the image - hash combination is already existend. If you can ensure,
    # that you run the test the first time and the database is empty, don't use the flag because
    # the execution time will be way longer using it.
    for sPathToImageSet in aImagesets:
        sensbench.hash_imageset(sPathToImageSet, bAddSave=False)
