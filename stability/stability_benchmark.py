#!/usr/bin/env python3

import sys
sys.path.append('../common')

import util
import dbcon
import math
from multiprocessing.pool import ThreadPool


class StabilityBenchmark:

    def __init__(
        self,
        sBaseFolder="../data/stability_results/",
        sPathToDB="stability_results.db",
        aAttacks=[],
        aHashes=[],
        fnDeviation=None,
        lNumberOfThreads=4
    ):

        # create folders if not existent
        self.sBaseFolder = util.create_path(sBaseFolder)

        # set attack list
        self.aAttacks = aAttacks

        # set hash function list
        self.aHashes = aHashes

        # set deviation function
        self.fnDeviation = fnDeviation

        # set number of threads
        self.lNumberOfThreads = lNumberOfThreads

        # create db file if not existent
        self.sPathToDB = sBaseFolder + sPathToDB
        open(self.sPathToDB, 'a').close()
        dbData = dbcon.Dbcon(self.sPathToDB)
        sDbSchema = """
        BEGIN TRANSACTION;
        CREATE TABLE IF NOT EXISTS `tests` (
            `id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            `attack`	INTEGER NOT NULL,
            `hash_type`	INTEGER NOT NULL,
            `image`	INTEGER NOT NULL,
            `original_hash`	INTEGER NOT NULL,
            `attacked_hash`	INTEGER NOT NULL,
            `deviation_hash`	REAL,
            FOREIGN KEY(`image`) REFERENCES `images`(`id`) ON UPDATE CASCADE ON DELETE RESTRICT,
            FOREIGN KEY(`attack`) REFERENCES `attacks`(`id`) ON UPDATE CASCADE ON DELETE RESTRICT,
            FOREIGN KEY(`attacked_hash`) REFERENCES `hashes`(`id`) ON UPDATE CASCADE ON DELETE RESTRICT,
            FOREIGN KEY(`hash_type`) REFERENCES `hash_types`(`id`) ON UPDATE CASCADE ON DELETE RESTRICT,
            FOREIGN KEY(`original_hash`) REFERENCES `hashes`(`id`) ON UPDATE CASCADE ON DELETE RESTRICT,
            UNIQUE(`attack`,`hash_type`,`image`)  --ON CONFLICT REPLACE
        );
        CREATE TABLE IF NOT EXISTS `images` (
            `id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            `name`	TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS `hashes` (
            `id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            `hash_value`	NPARRAY NOT NULL
        );
        CREATE TABLE IF NOT EXISTS `hash_types` (
            `id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            `hash_fn`	TEXT NOT NULL,
            `hash_params`	TEXT NOT NULL,
            UNIQUE(`hash_fn`, `hash_params`)
        );
        CREATE TABLE IF NOT EXISTS `attacks` (
            `id`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            `attack_fn`	TEXT NOT NULL,
            `attack_params`	TEXT NOT NULL,
            UNIQUE(`attack_fn`, `attack_params`)
        );
        COMMIT;
        """
        dbData.execute_sql_query_manipulation_script(sDbSchema)

# ---- setter ------
    def set_attacks(self, aAttacks):
        """ set attacks list
            use the following structure:

            [
                (attackFn1, {"lFactor": 10}),
                (attackFn1, {"lFactor": 20}),
                (attackFn2, {"bFit": True})
            ]
        """
        self.aAttacks = aAttacks

    def set_hashes(self, aHashes):
        """ set hashes list
            use the following structure:
            [
                (hashFn1, {}),
                (hashFn1, {"lHashSize": 512}),
                (hashFn2, {})
            ]
        """
        self.aHashes = aHashes

    def set_deviation_fn(self, fnDeviation=None):
        """set the deviation function to calculate the deviation
           between the hash of the original image and the hash of
           the attacked image

           set it to None if you do not want to calculate the deviation
           in the later analysis phase
        """
        self.fnDeviation = fnDeviation

    def set_nr_of_threads(self, lNumberOfThreads):
        """set the number of threads used to run the test
        """
        self.lNumberOfThreads = lNumberOfThreads

# ----- getter ------
    def get_dbcon(self):
        """ get db connection object"""
        return dbcon.Dbcon(self.sPathToDB)

# ---- private runner --------
    def __run_test_on_single_image(self, sPathToImage):
        """ run the test on a single image """
        # connect to db
        dbData = self.get_dbcon()
        # get image name
        sImageName = sPathToImage.split("/")[-1]
        aOriginalImage = util.load_image(sPathToImage)
        # check if there was an image with this name before
        aDBOriginalImage = dbData.execute_sql_query_select(
            "SELECT id FROM images WHERE name = ? ;", (sImageName,))
        lOriginalImageId = None
        if aDBOriginalImage:
            # get id of image if already in db
            lOriginalImageId = aDBOriginalImage[0][0]
        else:
            # create image and get id of entry
            lOriginalImageId = dbData.execute_sql_query_manipulation(
                "INSERT INTO images (name) VALUES (?);", (sImageName,))
        for fnAttack, dicAttackParams in self.aAttacks:
            # check whether the attack is already defined
            sAttackName = fnAttack.__name__
            sAttackParameters = str(dicAttackParams)
            tpValues = (sAttackName, sAttackParameters)
            aDBAttack = dbData.execute_sql_query_select(
                "SELECT id FROM attacks WHERE attack_fn=? AND attack_params=?;", tpValues)
            lAttackId = None
            # create it if not existend yet
            if not aDBAttack:
                lAttackId = dbData.execute_sql_query_manipulation(
                    "INSERT INTO attacks (attack_fn, attack_params) VALUES (?, ?);", tpValues)
            else:
                lAttackId = aDBAttack[0][0]
            # apply attack on image
            try:
                aAttackedImage = fnAttack(aOriginalImage, **dicAttackParams)
            except:
                print(
                    "An error occurred applying attack %s ... skipping Attack" % sAttackName)
                continue
            for fnHash, dicHashParams in self.aHashes:
                # check whether hash type is already defined
                sHashName = fnHash.__name__
                sHashParameters = str(dicHashParams)
                tpValues = (sHashName, sHashParameters)
                aDBHashTypes = dbData.execute_sql_query_select(
                    "SELECT Id FROM hash_types WHERE hash_fn=? and hash_params=?;", tpValues)
                lHashTypeId = None
                if aDBHashTypes:
                    lHashTypeId = aDBHashTypes[0][0]
                else:
                    lHashTypeId = dbData.execute_sql_query_manipulation(
                        "INSERT INTO hash_types (hash_fn, hash_params) VALUES (?, ?);", tpValues)
                # hash the original image if not done yet
                lOriginalImageHashId = None
                aOriginalImageHash = None
                aDBOriginalImagehashId = dbData.execute_sql_query_select(
                    "SELECT t.original_hash, h.hash_value FROM tests as t INNER JOIN hashes as h on h.id = t.original_hash WHERE t.image=? and t.hash_type=?;", (lOriginalImageId, lHashTypeId))
                if aDBOriginalImagehashId:
                    lOriginalImageHashId = aDBOriginalImagehashId[0][0]
                    aOriginalImageHash = aDBOriginalImagehashId[0][1]
                else:
                    # print("\t\t\tapplying hashing on original image")
                    try:
                        aOriginalImageHash = fnHash(
                            aOriginalImage, **dicHashParams)
                    except:
                        print("An error occurred generating hash with %s ... Skipping hash function" %
                              sHashName)
                        continue

                    lOriginalImageHashId = dbData.execute_sql_query_manipulation(
                        "INSERT INTO hashes (hash_value) VALUES (?);", (aOriginalImageHash,))
                # hash the attacked image
                try:
                    aAttackedImageHash = fnHash(
                        aAttackedImage, **dicHashParams)
                except:
                    print("An error occurred generating hash with %s ... Skipping hash function" %
                          sHashName)
                    continue
                lAttackedImageHashId = dbData.execute_sql_query_manipulation(
                    "INSERT INTO hashes (hash_value) VALUES (?);", (aAttackedImageHash,))
                # calc deviation of whished
                dDeviation = None
                if self.fnDeviation:
                    dDeviation = self.fnDeviation(
                        aOriginalImageHash, aAttackedImageHash)
                # save the test
                tpValues = (lAttackId, lHashTypeId, lOriginalImageId,
                            lOriginalImageHashId, lAttackedImageHashId, dDeviation)
                # print(str(tpValues[:3]) + " | " + sAttackName +
                # sAttackParameters + " | " + sHashName + " | " + sImageName)
                dbData.execute_sql_query_manipulation(
                    "INSERT INTO tests (attack, hash_type, image, original_hash, attacked_hash, deviation_hash) VALUES (?, ?, ?, ?, ?, ?);", tpValues)


#------ run test function --------

    def run_test_on_images(self, sPathToImages):
        """ run stability test on given directory of images
            images with file extension [".png", ".bmp", ".jpg", ".jpeg", ".tiff"]
            are considered
        """

        aImagePathes = util.list_all_images_in_directory(sPathToImages)

        # error handling
        if not aImagePathes:
            raise Exception(
                "given path %s does not contain any images" % sPathToDB)
        if not self.aHashes:
            raise Exception("you forgot to set hashing functions")
        if not self.aAttacks:
            raise Exception("you forgot to set attacks")
        if not self.fnDeviation:
            raise Exception("you forgot to set a deviation function")
        if not self.lNumberOfThreads >= 1:
            raise Exception(
                "the numbers of threads you defined is invalid: %i" % lNumberOfThreads)

        # cerate thread for every image
        oPool = ThreadPool(processes=self.lNumberOfThreads)
        aTaskPoolThreads = []
        for sPathToImage in aImagePathes:
            pThread = oPool.apply_async(
                self.__run_test_on_single_image, (sPathToImage,))
            aTaskPoolThreads.append(pThread)

        # catch threads -- have no returns
        for i in range(len(aImagePathes)):
            aTaskPoolThreads[i].get()
