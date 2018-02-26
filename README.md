# MIHBS
MIHBS, the modular image hashing benchmarking system for python3 using pandas, OpenCV2 and seaborn. The benchmarking system is designed to run under Linux. We cannot guarantee that it will work on other operating systems.

MIHBS is a set of python3 scripts to test and compare perceptual image hashing algorithms in terms of robustness, sensitivity and an arbitrary application-set.

## Installation Guide
To use the basic benchmark system without the pre-implemented algorithms, install the following python3 modules:
```bash
pip3 install seaborn pandas matplotlib sklearn numpy blend_modes skimage
```

Additionally install OpenCV2 and its python-wrapper. In some distributions you can find a pip module called `opencv-python` or `opencv-contrib-python`. Some distributions offer OpenCV2 as own package in its repositories.

If you want to compare your implementation against some of the pre-implemented ones install the following additional libraries:
```bash
pip3 install scipy PyWavelets
```

## Structure
There are three types of tests. Read the paper for a further explanation. Please note that term "robustness" used in the paper corresponds to "stability" used in the benchmark.

Every test setting has its own independent subdirectory. In each of the subdirectories `stability/`, `sensitivity/`, `application/` you can find a module named `testrunner.py` and one named `analyser.py`. Common modules like the database connection and the hashing algorithms you can find in the directory `common/`. The databases containing the test results will be placed in a subdirectory named `data/`. If you like, you can create a subdirectory `imagedatasets/` in the root of the project folder that contains the image data sets used for the tests.

## Execute the tests
The benchmark is separated into two phases. In the first phase the computationally complex tests are carried out. In oder to run the test, every test has an own `testrunner.py` script defining the test. The resulting data is written to local SQLite databases. Depending on the size of the test data sets or the complexity of the perceptual image hashing algorithms used, this phase can take a very long time. Therefore, it is intended that the first phase will be executed on a server. To take advantage of multicore processors, you can define how many threads should be used. For the server the following python modules are not necessary unless you use them in your own implementations: `seaborn`, `pandas`, `matplotlib`, `sklearn`. In oder to run the benchmark, copy the whole project including the image data sets to the server. The following subsections will explain how to configure the individual test settings.

### Stability tests

Open and edit `stability/testrunner.py`.

The attacks are defined as a list of functions taking an OpenCV 2 image as first parameter and an arbitrary dictionary of named parameters. The list is defined by `aAttacks`. To add your own attack first import the python module where your attack was defined. Entries of the attack list have to fit the following structure:

```python
(fnAttackFunction, {"parameter1": True, "parameter2": 42})
```

You can use python functions to create complex attack sets. In the following example we add three different noise attacks using the same set of parameters:

```python
# add noise
for dSigma in [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.11, 0.15, 0.2]:
    aTmpAttack = [
        (at.gauss_noise, {"dSigma": dSigma}),
        (at.speckle_noise, {"dSigma": dSigma}),
        (at.salt_and_pepper_noise, {"dAmount": dSigma})
    ]
    aAttacks.extend(aTmpAttack)
```

If you define attacks having no or only a few different parameter values, you can add your attack definitions to `aHandMadeAttacks`. Have a look in the code to find out what attacks are defined currently and which attacks are already implemented. You can find the attacks in `stability/attacks.py`. Feel free to add your own attacks and share them with the community by creating a pull-request.

The hashing algorithms are added to a list named `aHashes`. Import the python module of your own hashing algorithm and add a line to `aHashes`. Read the corresponding section to inform you how a hash algorithm has to be defined in order to be compatible to the benchmarking framework. You can test your hashing algorithm with different settings using the parameters:

```python
(ow.ownHashAlgo, {}),
(ow.ownHashAlgo, {"hashSize": 256}),
(ow.ownHashAlgo, {"hashSize": 512}),
```
If you really wish, you can define your own deviation function calculating the difference between two perceptual image hashes. Your function has to take two numpy boolean arrays of the same length and return a float. The normalized hamming distance is pre-implemented and will be used by default.

You can define the number of threads used in `lNumberOfThreads`. To avoid unintended side effects, every image of the stability image set will run on one single thread. Thus it makes no sense to define more threads than images in the stability data set.

Define the path to the stability data set in `sPathToImages"`.

After you have defined your testrun, you can save `stability/testrunner.py` and run it using `python3`:

```bash
cd stability
python3 testrunner.py
```

The results will be written to `data/stability_results/stability_results.db`. In order to analyze the data you only have to copy the database file back from the server.


### Sensitivity tests

Open and edit `sensitivity/testrunner.py`.

The hashing algorithms are added to a list named `aHashes`. Import the python module of your own hashing algorithm and add a line to `aHashes`. read the corresponding section to inform you how a hash algorithm has to be defined in order to be compatible to the benchmarking framework. You can test your hashing algorithm with different setting using the parameters:

```python
(ow.ownHashAlgo, {}),
(ow.ownHashAlgo, {"hashSize": 256}),
(ow.ownHashAlgo, {"hashSize": 512}),
```

You can define the number of threads used in `lNumberOfThreads`. In principle every image in the defined stability test data set could have its own thread. If you have very big data sets and a big amount of cores available on your server, feel free to define as much threads as cores or even more.

Define the path to the sensibility data sets by adding it to the list `aImagesets`.

```python
aImagesets = [
    "../imagedatasets/sensitivity/imgset1/",
    "../imagedatasets/sensitivity/imgset2/",
]
```

After you have defined your testrun you can save `sensitivity/testrunner.py` and run it using `python3`:

```bash
cd sensitivity
python3 testrunner.py
```

The results will be written to `data/sensitivity_results/sensitivity_results.db`. In order to analyze the data you only have to copy the database file back from the server.

If you already performed a run with a set of hashing algorithms and you want to add a new hashing algorithm to the test results, you can activate the `add-save` mode.

```python
for sPathToImageSet in aImagesets:
    sensbench.hash_imageset(sPathToImageSet, bAddSave=True)
```

In this mode, the benchmark will check for every single image whether the image - hash combination is already existent in teh database. If you can ensure, that you run the test the first time and the database is empty, don't use the flag because the execution will take way longer using the save mode.


### Application tests
Open and edit `application/testrunner.py`.

The definition of the testrunner is pretty equal to the sensitivity testrunner. Please read the section above.

The purpose of the application test is to test the perceptual image hashing algorithms with a set of real life application images. Read the paper to find out how we used the test. Our example covered print-scan images. You can feed the test with an arbitrary number of image sets. Please ensure that one image set contains the original images and that the filename of the images without file extension is part of the name of the remaining application images. Lets describe it with our example. We defined a reference image set in the directory `imagedatasets/application/ref/` containing the files:
 - baboon.tiff
 - jellybeans.tiff
 - lenna.tiff
 - peppers.tiff
 - sailboat.tiff
 - tiffany.tiff

Note, that the name of the folder will be taken as name for the image collection.

Then we printed and scanned the six images using different devices. One printer we used was named `D1`, one scanner `S2`. We added a second image set `imagedatasets/application/D1/` containing many images named like this examples:
```
D1_DC1_1200_150_2909_S1_P1_baboon.png
D1_DC1_1200_600_2909_S2_P1_jellybeans.png
D1_DC1_600_600_2909_S2_P1_sailboat.png
```
Later the analyzer will search for the substrings `baboon`, `jellybeans`, `lenna`, `peppers`, `sailboat`, `tiffany` to match the application images to the reference images. We used the names to decode information about the printer and scanner used, their print and scan resolution and the sort of paper used. If you define the naming scheme in a syntactically resolvable way, you can extract it later in the analysis as features.

After you have defined your testrun you can save `application/testrunner.py` and run it using `python3`:

```bash
cd application
python3 testrunner.py
```


## Analyze the results
The benchmark is separated into two phases. In the first phase the computationally complex tests are carried out and the results are written into databases. In the second phase the computed data can be analyzed on ordinary computers. In oder to perform the analyzes, every test has an own `analyser.py` script defining the analyzer steps. The first step of every analyzer is to parse the data from the SQLite databases to pandas data frames. Using pandas one can perform arbitrary analysis on the data sets. The following subsections describes the approach of the current analyzers.


### Stability tests
Open and edit `stability/analyser.py`.

First you can configure the paths `sStabilityResultsPath` where your plots and statistics should be saved to as well as the path to the database containing the results of the stability test run.

By changing the flag `bAllHashInOnePlot` accordingly you can decide to plot the results of all perceptual hashing algorithms in one diagram or to create a separated plot for each algorithm. Note that the same hash algorithm with different parameters will always be printed in on common plot.

Every attack needs to be handled accordingly. For the predefined attacks you can find predefined handlers for evaluating the data. If you define your own attack, don't forget to define an own handler and add it to `dicAttackHandler`. The handler definitions in the dictionary have the following form:

```python
dicAttackHandler = {
    ...
    "attack_function_name": (fnAttackDataHandler, {"parameter1": 42, "parameter2": True}),
    ...
}
```
The `"attack_function_name"` is the name of the python function your defined attack had. It will be used to filter the data, the handler gets passed. Have a look at the predefined handlers to understand how to deal with your data. It is important, that the first parameter is a pandas data frame. You can use the predefined plot and statistic functions or define your own evaluation for your attacks.

After you have defined your analysis you can save `stability/analyser.py` and run it using `python3`:

```bash
cd stability
python3 analyser.py
```

As far as you did not changed the destination, plots for each attack and if configured separated for each hash algorithm will be saved to `data/stability_results/*/plots`. You can find tables containing statistics for each attack and its parameters under `data/stability_results/*/stats`.

### Sensitivity tests

Open and edit `sensitivity/analyser.py`.

First you can configure the paths `sSensitivityTestResultsBasePath` where your plots and statistics should be saved to as well as the path to the database containing the results of the sensitivity test run.

The sensitivity test is designed to test one of your tested image sets. Thus you can specify which collection should be tested. The name of the collection is the name of the directory containing the images. Define the name in `sCollectionName`.

To calculate the sensitivity of your algorithms the test will split the image set randomly into a test data set and a reference data set. The every image from the test data set will be compared to every image in the reference data set. In `lTestDatasetSize` you can define the number of images the test set should have. Keep in mind that every image from the test set will be compared to every image in the reference set. The more test images you want to have the more time the test will consume. To keep the randomness fixed between multiple runs, you can define an integer as seed in `lRandomDataSplitSeed`.

The analyzer will plot the False Acceptance Rate of the algorithms. You can define the resolution of the error rate by defining the steps as array in `aThresholdSteps`.

After you have defined your analysis you can save `sensitivity/analyser.py` and run it using `python3`:

```bash
cd sensitivity
python3 analyser.py
```

As far as you did not changed the destination, plots for each attack and if configured separated for each hash algorithm will be saved to `data/stability_results/*/plots`. You can find tables containing statistics for each attack and its parameters under `data/stability_results/*/stats`.

### Application tests

Open and edit `application/analyser.py`.

First you can configure the paths `sApplicationTestResultBasePath` where your plots and statistics should be saved to as well as the path to the database containing the results of the application test run.

As described before you have to supply a reference image set showing the images without attacks. Define the name of the dataset in `sReferenceCollectionName`.

The analyzer will plot the FAR and FRR of the algorithms. You can define the resolution of the error rate by defining the threshold steps as array in `aThresholdSteps`.

If you which to, you can add the Equal Error Rate line to the plots by setting `bERR` to `True`.

After the test results from the database are loaded to a pandas data frame you can add features to the data by unpicking the image names. Have a look at the function `add_image_name_information_to_dataframe()` to get a feeling how this could be implemented.

Besides FAR and FRR plots you can analyze your application data grouped by your own application features. Have a look at the function `extract_user_defined_stats` to get a feeling how a grouped statistic could be generated. Feel free to use the function `save_group_to_files` to export grouping statistics to txt, tex and csv.

After you have defined your analysis you can save `sensitivity/analyser.py` and run it using `python3`:

```bash
cd sensitivity
python3 analyser.py
```
As far as you did not changed the destination, plots for each attack and if configured separated for each hash algorithm will be saved to `data/application_results/*/plots`. You can find tables containing statistics for each attack and its parameters under `data/application_results/*/stats`.

## How to define new attacks
Feel free to define new attacks. You can find the yet implemented attacks in `stability/attacks.py`. You can add your attack there or create and import your own file accordingly. To be compatible, your attack hat so fulfill the following specifications:
 - the first parameter of the attack function is a OpenCV2 image as numpy array
 - all other parameters are named and offer default values
 - the attack returns a manipulated OpenCV2 image as numpy array

 Have a look at the pre-defined attacks to get a clue how to define new ones.

## How to define new hashing algorithms
The purpose of this benchmark is to enable you tu test your own implementation of a perceptual image hashing algorithm to other algorithms and to test it against a specific set of attacks and a real world application scenario. To make your algorithm compatible to the MIHBS benchmark your implementation has to follow the following specifications:
 - the first parameter of the hashing function is a OpenCV2 image as numpy array
 - the algorithm returns a flat numpy bool array that can have an arbitrary length

## Bugs and Participation
Please report bugs using the github issue tracker. Feel free to fork and adapt the project and create pull-request for new features.

