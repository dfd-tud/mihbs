#!/usr/bin/env python3
import sys
sys.path.append('../common')

import util
import deviation
import dbcon
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# -------- helper functions -------


def convert_db_to_pandas(sPathToDB="../data/application_results/application_results.db"):
    # check if db is existend
    if not util.check_if_file_exists(sPathToDB):
        raise Exception("database %s is not existend" % sPathToDB)
    dbData = dbcon.Dbcon(sPathToDB)

    # get all tests and parse it to panda dataframe
    oApplicationTestData = pd.read_sql_query(
        "SELECT i.name as 'image', c.name as 'collection', h.hash, (ht.name || ' ' || ht.params) as 'hashalgorithm'  FROM images as i INNER JOIN collections as c on c.id = i.collection_id INNER JOIN images_hashes as ih on ih.image_id = i.id INNER JOIN hashes as h on h.id = ih.hash_id INNER JOIN hash_types as ht on ht.id = h.hash_type_id;", dbData.con)

    return oApplicationTestData


def get_deviations_to_original(oPandasData, sReferenceCollectionName):
    oPandasFullHashTypeDataset = None
    for sHashType in oPandasData.hashalgorithm.unique():
        for sImageBaseName in oPandasData[oPandasData.collection == sReferenceCollectionName].image.unique():
            sImageBaseName = sImageBaseName.split(".")[-2]
            oPandasHashData = oPandasData[(oPandasData.hashalgorithm == sHashType) & (
                oPandasData.image.str.contains(sImageBaseName))]
            oRefHash = oPandasHashData[oPandasHashData.collection ==
                                       sReferenceCollectionName].hash.values[0]
            oPandasHashData = oPandasHashData[oPandasHashData.collection !=
                                              sReferenceCollectionName]
            oPandasHashData["deviation"] = oPandasHashData.apply(
                lambda row: deviation.hamming_distance(row["hash"], oRefHash), axis=1)
            if isinstance(oPandasFullHashTypeDataset, pd.DataFrame):
                oPandasFullHashTypeDataset = pd.concat(
                    [oPandasFullHashTypeDataset, oPandasHashData])
            else:
                oPandasFullHashTypeDataset = oPandasHashData

    return oPandasFullHashTypeDataset


def get_deviations_to_notoriginal(oPandasData, sReferenceCollectionName):
    """ used for FAR test """
    oPandasFullHashTypeDataset = None
    for sHashType in oPandasData.hashalgorithm.unique():
        for i, sImageBaseName in enumerate(oPandasData[oPandasData.collection == sReferenceCollectionName].image.unique()):
            sImageBaseName = sImageBaseName.split(".")[-2]
            oPandasHashData = oPandasData[(oPandasData.hashalgorithm == sHashType) & (
                oPandasData.image.str.contains(sImageBaseName))]
            # choose a deterministic ref image that is not the original
            np.random.seed(i)
            oRefHash = np.random.choice(oPandasData[(oPandasData.collection == sReferenceCollectionName) & (oPandasData.hashalgorithm == sHashType) & (
                ~oPandasData.image.str.contains(sImageBaseName))].hash.values)
            oPandasHashData = oPandasHashData[oPandasHashData.collection !=
                                              sReferenceCollectionName]
            oPandasHashData["deviation"] = oPandasHashData.apply(
                lambda row: deviation.hamming_distance(row["hash"], oRefHash), axis=1)
            if isinstance(oPandasFullHashTypeDataset, pd.DataFrame):
                oPandasFullHashTypeDataset = pd.concat(
                    [oPandasFullHashTypeDataset, oPandasHashData])
            else:
                oPandasFullHashTypeDataset = oPandasHashData

    return oPandasFullHashTypeDataset


def plot_metrics(oPandasFRData, oPandasFAData, bERR=False):
    """ calculate and plot FAR and FRR
        oPandasFRData is the deviation by threshold to the original images
        oPandasFAData is the deviation by threshold to not the original images
    """

    sPlotPath = sApplicationTestResultBasePath + "plots/"
    util.create_path(sPlotPath)

    for sHashType in oPandasFAData["hashalgorithm"].unique():
        aFAR = [oPandasFAData[(oPandasFAData["hashalgorithm"] == sHashType) & (oPandasFAData["deviation"] <= i)]["image"].count(
        ) for i in aThresholdSteps] / oPandasFAData[oPandasFAData["hashalgorithm"] == sHashType]["deviation"].count()
        aFRR = (oPandasFRData[oPandasFRData["hashalgorithm"] == sHashType]["deviation"].count() - [oPandasFRData[(oPandasFRData["hashalgorithm"] == sHashType) & (
            oPandasFRData["deviation"] <= i)]["image"].count() for i in aThresholdSteps]) / oPandasFRData[oPandasFRData["hashalgorithm"] == sHashType]["deviation"].count()
        oPandasFAR = pd.DataFrame(
            {"Threshold": aThresholdSteps, "Errorrate": aFAR, "Type": ["FAR"] * len(aThresholdSteps)})
        oPandasFRR = pd.DataFrame(
            {"Threshold": aThresholdSteps, "Errorrate": aFRR, "Type": ["FRR"] * len(aThresholdSteps)})
        oPandasEERData = pd.concat([oPandasFAR, oPandasFRR])
        # pdEERData = pd.DataFrame({"Threshold": aTick, "FAR":aFAR, "FRR":
        # aFRR})
        oPandasEERData['subject'] = 0

        # calc less FAR FRR value
        lFARNotNullValueX = aThresholdSteps[np.argmax(np.array(aFAR) > 0) - 1]
        lFRRNotNullValueX = aThresholdSteps[np.argmax(np.array(aFRR) == 0)]
        if bERR:
            # calc ERR
            lMinDistancePosition = np.argmin((np.abs(aFRR - aFAR)))
            lERRValueX = aThresholdSteps[lMinDistancePosition]
            dErrorrateAtEER = np.mean(
                [aFAR[lMinDistancePosition], aFRR[lMinDistancePosition]])

            print("ERR: %f" % lERRValueX)
            print("Value: %f" % dErrorrateAtEER)

        plt.clf()
        sTitle = sHashType
        sb.set_style("whitegrid")
        oSeabornPlot = sb.tsplot(time="Threshold", value="Errorrate",
                                 condition="Type", unit="subject", interpolate=True, data=oPandasEERData)
        oSeabornPlot.set(title=sTitle)
        oSeabornPlot.set(xlabel="Threshold")
        oSeabornPlot.set(ylabel="Errorrate")
        oSeabornPlot.set(xticks=np.arange(0, 1.01, 0.1))
        oSeabornPlot.set(yticks=np.arange(0, 1.01, 0.1))
        oSeabornPlot.set(ylim=(0, 1))
        # add not zero lines
        plt.axvline(x=lFARNotNullValueX, color='#1470b0', linestyle="--")
        plt.axvline(x=lFRRNotNullValueX, color='#ff8c27', linestyle="--")
        if bERR:
            # add ERR line
            plt.axvline(x=lERRValueX, color='r', linestyle="-")
            # plt.axhline(y=dErrorrateAtEER, color='r', linestyle="-")
        sFileNameSafeTitle = util.format_filename(sTitle)
        oSeabornPlot.get_figure().savefig(sPlotPath + sFileNameSafeTitle + ".png")
        plt.clf()


def save_group_to_files(oPandasGroup, sTargetPath, sFileBaseName):
    """ calc min, mean, max for panda grouping and save to txt, csv and tex"""
    util.create_path(sTargetPath)
    oPandasGroup = oPandasGroup.agg(
        {"min": np.min, "mean": np.mean, "max": np.max})
    with open(sTargetPath + sFileBaseName + ".txt", "w") as file:
        oPandasGroup.to_string(file)
    with open(sTargetPath + sFileBaseName + ".tex", "w") as file:
        oPandasGroup.to_latex(file)
    with open(sTargetPath + sFileBaseName + ".csv", "w") as file:
        oPandasGroup.to_csv(file)
#---------- user defined functions ----------------


def add_image_name_information_to_dataframe(oPandasDataframe):

    aSplitNameArray = []
    for index, row in oPandasDataframe.iterrows():
        aSplitNameArray.append(row["image"].split("_"))

    pdDataFrameImageNamesSplit = pd.DataFrame({"split": aSplitNameArray})

    oPandasDataframe["printer"] = pdDataFrameImageNamesSplit.apply(
        lambda row: row["split"][0] if len(row["split"]) > 1 else np.NaN, axis=1)

    oPandasDataframe["printer_resolution"] = pdDataFrameImageNamesSplit.apply(
        lambda row: row["split"][2] if len(row["split"]) > 1 else np.NaN, axis=1)

    oPandasDataframe["scanner_resolution"] = pdDataFrameImageNamesSplit.apply(
        lambda row: row["split"][3] if len(row["split"]) > 1 else np.NaN, axis=1)

    oPandasDataframe["scanner"] = pdDataFrameImageNamesSplit.apply(
        lambda row: row["split"][5] if len(row["split"]) > 1 else np.NaN, axis=1)

    oPandasDataframe["paper"] = pdDataFrameImageNamesSplit.apply(
        lambda row: row["split"][6] if len(row["split"]) > 1 else np.NaN, axis=1)

    oPandasDataframe["special"] = pdDataFrameImageNamesSplit.apply(
        lambda row: row["split"][7] if len(row["split"]) > 8 else np.NaN, axis=1)

    return oPandasDataframe


def extract_user_defined_stats(oPandasDeviationsToOriginal):
    sApplicationTestResultStatsBasePath = sApplicationTestResultBasePath + "stats/"

    util.create_path(sApplicationTestResultStatsBasePath)

    # base - cumulated over all printers, resolutions, etc.
    save_group_to_files(oPandasDeviationsToOriginal.groupby(
        ["hashalgorithm"])["deviation"], sApplicationTestResultStatsBasePath + "original/", "base")

    # printer
    save_group_to_files(oPandasDeviationsToOriginal[oPandasDeviationsToOriginal["special"].isnull()].groupby(
        ["hashalgorithm", "printer"])["deviation"], sApplicationTestResultStatsBasePath + "original/", "printer")

    # printer - resolution
    save_group_to_files(oPandasDeviationsToOriginal[oPandasDeviationsToOriginal["special"].isnull()].groupby(
        ["hashalgorithm", "printer", "printer_resolution"])["deviation"], sApplicationTestResultStatsBasePath + "original/", "printer_resolution")

    # printer - resolution - (clustered)
    save_group_to_files(oPandasDeviationsToOriginal[oPandasDeviationsToOriginal["special"].isnull()].groupby(
        ["printer", "printer_resolution"])["deviation"], sApplicationTestResultStatsBasePath + "original/", "printer_resolution_clustered")

    # printer - special
    save_group_to_files(oPandasDeviationsToOriginal[oPandasDeviationsToOriginal["printer"] != "D1"].fillna("none").groupby(
        ["hashalgorithm", "printer", "special"])["deviation"], sApplicationTestResultStatsBasePath + "original/", "printer_special")

    # paper
    save_group_to_files(oPandasDeviationsToOriginal[oPandasDeviationsToOriginal["special"].isnull()].groupby(
        ["hashalgorithm", "paper"])["deviation"], sApplicationTestResultStatsBasePath + "original/", "paper")

    # scanner - resolution
    save_group_to_files(oPandasDeviationsToOriginal.groupby(
        ["hashalgorithm", "scanner_resolution"])["deviation"], sApplicationTestResultStatsBasePath + "original/", "scanner_resolution")

    # ----------------------------------------------------


if __name__ == "__main__":
    # ---------- config ----------------------

    # set name reference collection
    sReferenceCollectionName = "ref"

    # define the steps for the calculation of the error rate
    # and the plot (min, max + step, step)
    aThresholdSteps = np.arange(0, 1.01, 0.01)

    # define path for plots and starts
    sApplicationTestResultBasePath = "../data/application_results/"

    # set size of the plot
    # set figure size
    plt.figure(num=None, figsize=(7, 6), dpi=100,
               facecolor='w', edgecolor='k')

    # plot ERR
    # NOTE: it is not very clear where to place the ERR
    # if the err is placed at a strange position you can
    # deactivate the EER plot by setting this flag to False
    bERR = False

    # ----------------------------------------

    # get all data in pandas dataframe
    oApplicationTestData = convert_db_to_pandas()

    ###### USER SPECIFIC#####################
    # define custom feature generators here
    oApplicationTestData = add_image_name_information_to_dataframe(
        oApplicationTestData)

    #########################################

    # calculate deviations to original
    oPandasDeviationsToOriginal = get_deviations_to_original(
        oApplicationTestData, sReferenceCollectionName)

   # calculate deviations to not original
    oPandasDeviationsToNonoriginal = get_deviations_to_notoriginal(
        oApplicationTestData, sReferenceCollectionName)

    plot_metrics(oPandasDeviationsToOriginal,
                 oPandasDeviationsToNonoriginal, bERR=bERR)

    ####### USER SPECIFIC ####################
    # define your custom analyser functions
    extract_user_defined_stats(oPandasDeviationsToOriginal)
    ##########################################
