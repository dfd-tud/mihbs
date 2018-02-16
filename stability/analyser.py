#!/usr/bin/env python3
import sys
sys.path.append('../common')

import util
import dbcon
import ast
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# --------- helper functions -------

def convert_db_to_pandas(sPathToDB="./data/stability_results/stability_results.db"):
    # check if db is existend
    if not util.check_if_file_exists(sPathToDB):
        raise Exception("database %s is not existend" % sPathToDB)
    dbData = dbcon.Dbcon(sPathToDB)

    # get all tests and parse it to panda dataframe
    oStabilityTestData = pd.read_sql_query(
        "SELECT t.id, t.image, t.deviation_hash, ht.hash_fn, ht.hash_params , a.attack_fn, a.attack_params, i.name as 'image_name' FROM tests as t INNER JOIN attacks as a on a.id = t.attack INNER JOIN hash_types as ht on ht.id = t.hash_type INNER JOIN images as i ON i.id = t.image;", dbData.con)

    oStabilityTestData["hashalgorithm"] = oStabilityTestData.apply(
        lambda row: row["hash_fn"] + " " + row["hash_params"], axis=1)

    return oStabilityTestData


def get_list_of_attacks(oPandasData):
    """ returns a list of attacks when a pandas dataframe if given """
    return oStabilityTestData.attack_fn.unique()


# ----------- diagram functions ----------------------------#

def plot_single_parameter_sortable(sImageName,
                                   oPandasData,
                                   sXColumnName,
                                   sYColumnName,
                                   sCategoryColumnName,
                                   sBasePath,
                                   sUnitColumnName="image",
                                   sDiagramTitle=None,
                                   sXLabel=None,
                                   sYLabel=None,
                                   aXTicks=None,
                                   lConfidenceInterval=95,
                                   bInterpolate=True,
                                   aYLim=None):
    """plot the mean values and the confidence interval over the unit column for each category (mostly hashes)"""
    plt.clf()
    # sb.set_style("whitegrid")
    oSeabornPlot = sb.tsplot(time=sXColumnName, value=sYColumnName, unit=sUnitColumnName, condition=sCategoryColumnName, ci=lConfidenceInterval, interpolate=bInterpolate,
                             data=oPandasData)
    if sDiagramTitle:
        oSeabornPlot.set(title=sDiagramTitle)
    if sXLabel:
        oSeabornPlot.set(xlabel=sXLabel)
    if sYLabel:
        oSeabornPlot.set(ylabel=sYLabel)
    if not (aXTicks is None):
        oSeabornPlot.set(xticks=aXTicks)
    if not (aYLim is None):
        oSeabornPlot.set(ylim=aYLim)
    sTargetPath = sStabilityResultsPath + sBasePath + sPlotSubpath
    util.create_path(sTargetPath)
    oSeabornPlot.get_figure().savefig(sTargetPath + sImageName)
    plt.clf()


def plot_single_parameter_sortable_for_each_hash(sImageBaseName,
                                                 sImageExtension,
                                                 oPandasData,
                                                 sXColumnName,
                                                 sYColumnName,
                                                 sCategoryColumnName,
                                                 sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle=None,
                                                 sXLabel=None,
                                                 sYLabel=None,
                                                 aXTicks=None,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True):
    """for each hashfunction: plot the mean values and the confidence interval over the unit column for each category (mostly hashes)"""
    dYMax = np.percentile(oPandasData[sYColumnName].values, 99)
    aYLim = (0, dYMax)
    if bAllHashInOnePlot:
        plot_single_parameter_sortable(sImageBaseName + sImageExtension,
                                       oPandasData,
                                       sXColumnName,
                                       sYColumnName,
                                       sCategoryColumnName,
                                       sBasePath,
                                       sUnitColumnName,
                                       sDiagramTitle,
                                       sXLabel,
                                       sYLabel,
                                       aXTicks,
                                       lConfidenceInterval,
                                       bInterpolate,
                                       aYLim=aYLim)
    else:
        for sHashFunction in oPandasData["hash_fn"].unique():
            plot_single_parameter_sortable(sImageBaseName + "__" + sHashFunction + sImageExtension,
                                           oPandasData[oPandasData["hash_fn"]
                                                       == sHashFunction],
                                           sXColumnName,
                                           sYColumnName,
                                           sCategoryColumnName,
                                           sBasePath,
                                           sUnitColumnName,
                                           sDiagramTitle,
                                           sXLabel,
                                           sYLabel,
                                           aXTicks,
                                           lConfidenceInterval,
                                           bInterpolate,
                                           aYLim=aYLim)


def plot_single_parameter_categorical(sImageName,
                                      oPandasData,
                                      sXColumnName,
                                      sYColumnName,
                                      sCategoryColumnName,
                                      sBasePath,
                                      sDiagramTitle=None,
                                      sXLabel=None,
                                      sYLabel=None,
                                      aXTicks=None,
                                      lConfidenceInterval=95,
                                      aYLim=None):
    """plot the mean values and the confidence interval over whole dataFrame for each category (mostly hashes)"""
    plt.clf()
    # i have to filter the columns here because this plot can not deal with
    # multiple ids etc.
    oPandasData = oPandasData[[sXColumnName,
                               sYColumnName, sCategoryColumnName]]
    oSeabornPlot = sb.barplot(x=sXColumnName, y=sYColumnName,
                              hue=sCategoryColumnName, ci=lConfidenceInterval, capsize=.2,  data=oPandasData)
    if sDiagramTitle:
        oSeabornPlot.set(title=sDiagramTitle)
    if sXLabel:
        oSeabornPlot.set(xlabel=sXLabel)
    if sYLabel:
        oSeabornPlot.set(ylabel=sYLabel)
    if not (aXTicks is None):
        oSeabornPlot.set(xticks=aXTicks)
    if not (aYLim is None):
        oSeabornPlot.set(ylim=aYLim)
    sTargetPath = sStabilityResultsPath + sBasePath + sPlotSubpath
    util.create_path(sTargetPath)
    oSeabornPlot.get_figure().savefig(sTargetPath + sImageName)
    plt.clf()


def plot_single_parameter_categorical_for_each_hash(sImageBaseName,
                                                    sImageExtension,
                                                    oPandasData,
                                                    sXColumnName,
                                                    sYColumnName,
                                                    sCategoryColumnName,
                                                    sBasePath,
                                                    sDiagramTitle=None,
                                                    sXLabel=None,
                                                    sYLabel=None,
                                                    aXTicks=None,
                                                    lConfidenceInterval=95):
    """for each hashfunction: plot the mean values and the confidence interval over the pandas dataframe for each category (mostly hashes)"""
    dYMax = np.percentile(oPandasData[sYColumnName].values, 99)
    aYLim = (0, dYMax)
    if bAllHashInOnePlot:
        plot_single_parameter_categorical(sImageBaseName + sImageExtension,
                                          oPandasData,
                                          sXColumnName,
                                          sYColumnName,
                                          sCategoryColumnName,
                                          sBasePath,
                                          sDiagramTitle,
                                          sXLabel,
                                          sYLabel,
                                          aXTicks,
                                          lConfidenceInterval,
                                          aYLim=aYLim)
    else:
        for sHashFunction in oPandasData["hash_fn"].unique():
            plot_single_parameter_categorical(sImageBaseName + "__" + sHashFunction + sImageExtension,
                                              oPandasData[oPandasData["hash_fn"]
                                                          == sHashFunction],
                                              sXColumnName,
                                              sYColumnName,
                                              sCategoryColumnName,
                                              sBasePath,
                                              sDiagramTitle,
                                              sXLabel,
                                              sYLabel,
                                              aXTicks,
                                              lConfidenceInterval,
                                              aYLim=aYLim)

# --------------------- data stats generators ---------------------- #


def calc_values_single_parameter_for_each_hash(sFileBaseName, oPandasData, sXColumnName, sYColumnName, sCategoryColumnName, sBasePath, sFileExtension=".txt"):
    """ save the min, mean and max values to file grouped by every hash algorithm and parameter value"""
    sTargetPath = sStabilityResultsPath + sBasePath + sStatsSubpath
    util.create_path(sTargetPath)
    with open(sTargetPath + sFileBaseName + sFileExtension, 'w') as file:
        oPandasData.groupby([sCategoryColumnName, sXColumnName])[sYColumnName].agg(
            {"min": np.min, "mean": np.mean, "max": np.max}).to_string(file)


def calc_values_multiple_parameter_for_each_hash(sFileBaseName, oPandasData, sYColumnName, sCategoryColumnName, sBasePath, sFileExtension=".txt"):
    """ save the min, mean and max values to file grouped by every hash algorithm and parameters of attack 1 and 2"""
    sTargetPath = sStabilityResultsPath + sBasePath + sStatsSubpath
    util.create_path(sTargetPath)
    oGroupDF = oPandasData.groupby([sCategoryColumnName, "attack1", "params1", "attack2", "params2"])[sYColumnName].agg(
        {"min": np.min, "mean": np.mean, "max": np.max})
    with open(sTargetPath + sFileBaseName + sFileExtension, 'w') as file:
        oGroupDF.to_string(file)
    with open(sTargetPath + sFileBaseName + ".tex", 'w') as file:
        oGroupDF.to_latex(file)
    with open(sTargetPath + sFileBaseName + ".csv", 'w') as file:
        oGroupDF.to_csv(file)


# ------------------------- attack handlers --------------------------
def scale_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # calculate x and y parameters
    oPandasData["paramY"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["lScaleFactorY"])
    # TODO: change the typo in the attacks and here and in stability test
    oPandasData["paramX"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["lScalefactorX"])

    # first progress with uniform scale
    oScaleUniform = oPandasData[oPandasData["paramX"] == oPandasData["paramY"]]

    aScaleTypes = [
        ("scale_uniform", "scale (uniform)", "paramX", "scale factor x and y",
         oPandasData[oPandasData["paramX"] == oPandasData["paramY"]]),
        ("scale_nonuniform_x", "scale (non-uniform, x-axis)", "paramX", "scale factor  x",
         oPandasData[oPandasData["paramY"] == 1]),
        ("scale_nonuniform_y", "scale (non-uniform, x-axis)", "paramY", "scale factor  y",
         oPandasData[oPandasData["paramX"] == 1])
    ]
    for sFileBaseName, sDiagramTitle, sXColumnName, sXLabel, oData in aScaleTypes:
        # plot uniform scale data
        plot_single_parameter_sortable_for_each_hash(sImageBaseName=sFileBaseName,
                                                     sImageExtension=".png",
                                                     oPandasData=oData,
                                                     sXColumnName=sXColumnName,
                                                     sYColumnName=sYColumnName,
                                                     sCategoryColumnName="hashalgorithm",
                                                     sBasePath=sBasePath,
                                                     sUnitColumnName="image",
                                                     sDiagramTitle=sDiagramTitle,
                                                     sXLabel=sXLabel,
                                                     sYLabel=sYLabel,
                                                     lConfidenceInterval=95,
                                                     bInterpolate=True)
        calc_values_single_parameter_for_each_hash(
            sFileBaseName, oPandasData=oData, sXColumnName=sXColumnName, sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)

    # continue here with non uniform values
    oNonuniformScale = oPandasData[(oPandasData["paramX"] != oPandasData["paramY"]) & (
        oPandasData["paramX"] != 1) & (oPandasData["paramY"] != 1)]

    # create parameter
    oNonuniformScale["parameter"] = oNonuniformScale["attack_params"].map(
        lambda a: "(" + str(ast.literal_eval(a)["lScalefactorX"]) + ", " + str(ast.literal_eval(a)["lScaleFactorY"]) + ")")

    plot_single_parameter_categorical_for_each_hash(sImageBaseName="scale_nonuniform",
                                                    sImageExtension=".png",
                                                    oPandasData=oNonuniformScale,
                                                    sXColumnName="parameter",
                                                    sYColumnName=sYColumnName,
                                                    sCategoryColumnName="hashalgorithm",
                                                    sBasePath=sBasePath,
                                                    sDiagramTitle="scale (non-uniform)",
                                                    sXLabel="scalefactor (x, y)",
                                                    sYLabel=sYLabel,
                                                    lConfidenceInterval=95)
    calc_values_single_parameter_for_each_hash(
        "scale_nonuniform", oPandasData=oNonuniformScale, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def rotation_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):
    """process rotation as single unsegmented attack """
    # add rotation angle as feature
    oPandasData["parameter"] = oPandasData.apply(
        lambda row: ast.literal_eval(row["attack_params"])["dRotationAngle"], axis=1)

    aRotationTypes = [
        ("rotation_scale", "rotation (scaled)",
         oPandasData[oPandasData["attack_params"].str.contains("'bFit': True")]),
        ("rotation_noscale", "rotation (unscaled)",
         oPandasData[oPandasData["attack_params"].str.contains("'bFit': False")])
    ]

    for sFileBaseName, sDiagramTitle, oData in aRotationTypes:
        plot_single_parameter_sortable_for_each_hash(sImageBaseName=sFileBaseName,
                                                     sImageExtension=".png",
                                                     oPandasData=oData,
                                                     sXColumnName="parameter",
                                                     sYColumnName=sYColumnName,
                                                     sCategoryColumnName="hashalgorithm",
                                                     sBasePath=sBasePath,
                                                     sUnitColumnName="image",
                                                     sDiagramTitle=sDiagramTitle,
                                                     sXLabel="angle of rotation",
                                                     sYLabel=sYLabel,
                                                     lConfidenceInterval=95,
                                                     aXTicks=np.arange(
                                                         0, 361, 30),
                                                     bInterpolate=True)
        calc_values_single_parameter_for_each_hash(
            sFileBaseName, oPandasData=oData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def rotation_cropped_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):
    """process rotation as single unsegmented attack """

    # add rotation angle as feature
    oPandasData["parameter"] = oPandasData.apply(
        lambda row: ast.literal_eval(row["attack_params"])["dRotationAngle"], axis=1)

    plot_single_parameter_sortable_for_each_hash(sImageBaseName="rotation_cropped",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="rotation (cropped)",
                                                 sXLabel="angle of rotation",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 aXTicks=np.arange(
                                                     0, 361, 30),
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "rotation_cropped", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def crop_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):
    # add params
    oPandasData["paramTop"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["tpSlice"][0])
    oPandasData["paramLeft"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["tpSlice"][1])
    oPandasData["paramButtom"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["tpSlice"][2])
    oPandasData["paramRight"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["tpSlice"][3])

    # handle uniform crops
    oCropUniform = oPandasData[(oPandasData["paramTop"] == oPandasData["paramLeft"]) & (
        oPandasData["paramTop"] == oPandasData["paramButtom"]) & (oPandasData["paramTop"] == oPandasData["paramRight"])]

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="crop_uniform",
                                                 sImageExtension=".png",
                                                 oPandasData=oCropUniform,
                                                 sXColumnName="paramTop",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="crop (uniform)",
                                                 sXLabel="crop factor in % (all sides)",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "crop_uniform", oPandasData=oCropUniform, sXColumnName="paramTop", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)

    # handle nonuniform cropping
    oCropNonUniform = oPandasData[(oPandasData["paramTop"] != oPandasData["paramLeft"]) | (
        oPandasData["paramTop"] != oPandasData["paramButtom"]) | (oPandasData["paramTop"] != oPandasData["paramRight"])]

    # create parameter
    oCropNonUniform["parameter"] = oCropNonUniform["attack_params"].map(
        lambda a: str(ast.literal_eval(a)["tpSlice"]))

    plot_single_parameter_categorical_for_each_hash(sImageBaseName="crop_nonuniform",
                                                    sImageExtension=".png",
                                                    oPandasData=oCropNonUniform,
                                                    sXColumnName="parameter",
                                                    sYColumnName=sYColumnName,
                                                    sCategoryColumnName="hashalgorithm",
                                                    sBasePath=sBasePath,
                                                    sDiagramTitle="crop (non-uniform)",
                                                    sXLabel="crop factor in % (top, left, buttom, right)",
                                                    sYLabel=sYLabel,
                                                    lConfidenceInterval=95)
    calc_values_single_parameter_for_each_hash(
        "crop_nonuniform", oPandasData=oCropNonUniform, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def shift_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["lPixles"])
    aShiftTypes = [
        ("shift_vertical", "shift (vertical)", "shift vertical in pixel",
         oPandasData[oPandasData["attack_fn"].str.contains("_vertical")]),
        ("shift_horizontal", "shift (horizontal)", "shift horizontal in pixel",
         oPandasData[oPandasData["attack_fn"].str.contains("_horizontal")])
    ]
    # check what kind of shift was given
    lIndex = 0
    if oPandasData.attack_fn.unique()[0] == "shift_horizontal":
        lIndex = 1
    sFileBaseName, sDiagramTitle, sXLabel, oData = aShiftTypes[lIndex]
    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName=sFileBaseName,
                                                 sImageExtension=".png",
                                                 oPandasData=oData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle=sDiagramTitle,
                                                 sXLabel=sXLabel,
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        sFileBaseName, oPandasData=oData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def flip_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: "horizontal" if a.find("{'bVertical': False}") >= 0 else "vertical")

    plot_single_parameter_categorical_for_each_hash(sImageBaseName="flip",
                                                    sImageExtension=".png",
                                                    oPandasData=oPandasData,
                                                    sXColumnName="parameter",
                                                    sYColumnName=sYColumnName,
                                                    sCategoryColumnName="hashalgorithm",
                                                    sBasePath=sBasePath,
                                                    sDiagramTitle="flip",
                                                    sXLabel="direction",
                                                    sYLabel=sYLabel,
                                                    lConfidenceInterval=95)
    calc_values_single_parameter_for_each_hash(
        "flip", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def contrast_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["lContrast"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="contrast",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="contrast",
                                                 sXLabel="c",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "contrast", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def gamma_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["dGamma"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="gamma",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="gamma adjustment",
                                                 sXLabel="$\gamma$",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "gamma", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def median_filter_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["lKernelSize"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="median_filter",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="median filter",
                                                 sXLabel="k",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "median_filter", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def gauss_filter_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["lSigma"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="gauss_filter",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="gaussian filter",
                                                 sXLabel="$\sigma$",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "gauss_filter", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def jpeg_compression_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["lJPEGQuality"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="jpeg",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="JPEG compression",
                                                 sXLabel="quality (in %)",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "jpeg", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def gauss_noise_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["dSigma"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="gauss_noise",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="gaussian noise",
                                                 sXLabel="percent of affected pixel",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "gauss_noise", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def speckle_noise_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["dSigma"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="speckle_noise",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="speckle noise",
                                                 sXLabel="percent of affected pixel",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "speckle_noise", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def salt_and_pepper_noise_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["dAmount"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="salt_and_pepper_noise",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="salt and pepper noise",
                                                 sXLabel="percent of affected pixel",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "salt_and_pepper_noise", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def brightness_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: ast.literal_eval(a)["lBrightness"])

    # plot uniform scale data
    plot_single_parameter_sortable_for_each_hash(sImageBaseName="brightness",
                                                 sImageExtension=".png",
                                                 oPandasData=oPandasData,
                                                 sXColumnName="parameter",
                                                 sYColumnName=sYColumnName,
                                                 sCategoryColumnName="hashalgorithm",
                                                 sBasePath=sBasePath,
                                                 sUnitColumnName="image",
                                                 sDiagramTitle="brightness adjustment",
                                                 sXLabel="h",
                                                 sYLabel=sYLabel,
                                                 lConfidenceInterval=95,
                                                 bInterpolate=True)
    calc_values_single_parameter_for_each_hash(
        "brightness", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)


def pattern_handler(oPandasData, sBasePath="basic/", sYColumnName="deviation_hash", sYLabel="mean Hamming distance (ci = 95%)"):
    # filter all pattern
    oPandasData = oPandasData[oPandasData["attack_fn"] == "blend_pattern"]

    # create parameter
    oPandasData["parameter"] = oPandasData["attack_params"].map(
        lambda a: "white paper" if a.find("array([[[252,") >= 0 else "recycled paper")

    plot_single_parameter_categorical_for_each_hash(sImageBaseName="pattern",
                                                    sImageExtension=".png",
                                                    oPandasData=oPandasData,
                                                    sXColumnName="parameter",
                                                    sYColumnName=sYColumnName,
                                                    sCategoryColumnName="hashalgorithm",
                                                    sBasePath=sBasePath,
                                                    sDiagramTitle="pattern blending",
                                                    sXLabel="pattern",
                                                    sYLabel=sYLabel,
                                                    lConfidenceInterval=95)
    calc_values_single_parameter_for_each_hash(
        "pattern", oPandasData=oPandasData, sXColumnName="parameter", sYColumnName=sYColumnName, sCategoryColumnName="hashalgorithm", sBasePath=sBasePath)

#----------------------- main function ----------------------------


if __name__ == "__main__":

    # -------- global config ----------------
    # NOTE: add your custom pathes here
    sStabilityResultsPath = "../data/stability_results/"
    sPlotSubpath = "plots/"
    sStatsSubpath = "stats/"

    sPathToDB = "../data/stability_results/stability_results.db"

    # set figure size
    plt.figure(num=None, figsize=(7, 6), dpi=100, facecolor='w', edgecolor='k')

    # plot all hashalgos in one plot
    bAllHashInOnePlot = True

    # NOTE: if you define a new attack, you have to implement aAttacks
    # handler that extracts the data and plots the charts needed
    dicAttackHandler = {
        "scale": (scale_handler, {}),
        "rotation": (rotation_handler, {}),
        "rotation_cropped": (rotation_cropped_handler, {}),
        "crop_percentage": (crop_handler, {}),
        "shift_vertical": (shift_handler, {}),
        "shift_horizontal": (shift_handler, {}),
        "flip": (flip_handler, {}),
        "contrast": (contrast_handler, {}),
        "gamma_adjustment": (gamma_handler, {}),
        "median_filter": (median_filter_handler, {}),
        "gaussian_filter": (gauss_filter_handler, {}),
        "jpeg_compression": (jpeg_compression_handler, {}),
        "gauss_noise": (gauss_noise_handler, {}),
        "speckle_noise": (speckle_noise_handler, {}),
        "salt_and_pepper_noise": (salt_and_pepper_noise_handler, {}),
        "brightness": (brightness_handler, {}),
        "blend_pattern": (pattern_handler, {}),
    }

    # ------------------------------------------------------

    # read database
    oStabilityTestData = convert_db_to_pandas(sPathToDB)

    # get list of attacks applied
    aAttacks = get_list_of_attacks(oStabilityTestData)

    # run handler for every attack defined
    for sAttack in aAttacks:
        if not(sAttack in dicAttackHandler.keys()):
            print("There is no handler for attack %s" % sAttack)
            continue
        fnAttackHandler, dicHandlerParams = dicAttackHandler[sAttack]
        # run attackhandler with params
        fnAttackHandler(
            oStabilityTestData[oStabilityTestData["attack_fn"] == sAttack], **dicHandlerParams)
