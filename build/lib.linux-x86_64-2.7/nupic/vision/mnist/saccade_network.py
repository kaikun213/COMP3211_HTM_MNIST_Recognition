# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import copy
import collections
import time

from PIL import ImageTk, ImageDraw, Image
import yaml

from htmresearch.regions.ColumnPoolerRegion import ColumnPoolerRegion
from htmresearch.regions.ExtendedTMRegion import ExtendedTMRegion
from nupic.engine import Network

from nupic.vision.regions.SaccadeSensor import SaccadeSensor
from nupic.vision.image import deserializeImage

SACCADES_PER_IMAGE_TRAINING = 60
SACCADES_PER_IMAGE_TESTING = 20
_SACCADE_SIZE = 7
_FOVEA_SIZE = 14
_MAX_DRIFT = -2
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

DEFAULT_IMAGESENSOR_PARAMS = {
    "width": 28,
    "height": 28,
    "mode": "bw",
    "background": 0,
    "explorer": yaml.dump(["RandomSaccade", {
        "replacement": False,
        "saccadeMin": _SACCADE_SIZE,
        "saccadeMax": _SACCADE_SIZE,
        "numSaccades": SACCADES_PER_IMAGE_TRAINING,
        "maxDrift": _MAX_DRIFT,
        "seed": 1738}]),
    "postFilters": yaml.dump([["Resize", {
        "size": (_FOVEA_SIZE, _FOVEA_SIZE), "method": "center" }]]),
}

DEFAULT_SP_PARAMS = {
    "columnCount": 2048,
    "spatialImp": "cpp",
    "inputWidth": 784,
    "spVerbosity": 1,
    "synPermConnected": 0.2,
    "synPermActiveInc": 0.1,
    "synPermInactiveDec": 0.03,
    "seed": 1956,
    "numActiveColumnsPerInhArea": 40,
    "globalInhibition": 1,
    "potentialPct": 0.8,
    "boostStrength": 0.0,
}

DEFAULT_TM_PARAMS = {
    "implementation": "etm",
    "columnDimensions": 0,
#    "numberOfDistalInput": 0,
    "cellsPerColumn": 8,
    "initialPermanence": 0.4,
    "connectedPermanence": 0.5,
    "minThreshold": 20,
    "activationThreshold": 20,
#    "newSynapseCount": 50,
#    "newDistalSynapseCount": 50,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "maxSegmentsPerCell": 16,
    "learnOnOneCell": "true",
#    "learnDistalInputs": True,
#    "learnLateralConnections": False,
#    "globalDecay": 0,
#    "burnIn": 1,
#    "verbosity": 0,
    "apicalInputWidth": 2048,
}

DEFAULT_TP_PARAMS = {
    "learningMode": True,
    "cellCount": 2048,
    "inputWidth": 2048*8,
    "numOtherCorticalColumns": 0,
    "sdrSize": 40,
    "synPermProximalInc": 0.1,
    "synPermProximalDec": 0.001,
    "initialProximalPermanence": 0.6,
    "sampleSizeProximal": 20,
    "minThresholdProximal": 1,
    "connectedPermanenceProximal": 0.5,
    "synPermDistalInc": 0.1,
    "synPermDistalDec": 0.001,
    "initialDistalPermanence": 0.41,
    "sampleSizeDistal": 20,
    "activationThresholdDistal": 13,
    "connectedPermanenceDistal": 0.5,
    "seed": 42,
}

DEFAULT_CLASSIFIER_PARAMS = {
    "distThreshold": 0.000001,
    "maxCategoryCount": 10,
}



class SaccadeNetwork(object):
  """
  A HTM network structured as follows:
  SaccadeSensor (RandomSaccade) -> SP -> TM -> Classifier (KNN)
  """
  def __init__(self,
               networkName,
               trainingSet,
               testingSet,
               loggingDir=None,
               validationSet=None,
               detailedSaccadeWidth=IMAGE_WIDTH,
               detailedSaccadeHeight=IMAGE_HEIGHT,
               createNetwork=True):
    """
    :param str networkName: Where the network will be serialized/saved to
    :param str trainingSet: Path to set of images to train on
    :param str testingSet: Path to set of images to test
    :param str loggingDir: directory to store logged images in
      (note: no image logging if none)
    :param validationSet: (optional) Path to set of images to validate on
    :param int detailedSaccadeWidth: (optional) Width of detailed saccades to
      return from the runNetworkOneImage and testNetworkOneImage
    :param int detailedSaccadeHeight: (optional) Height of detailed saccades to
      return from the runNetworkOneImage and testNetworkOneImage
    :param bool createNetwork: If false, wait until createNet is manually
      called to create the network. Otherwise, create on __init__
    """
    self.loggingDir = loggingDir
    self.netFile = networkName
    self.trainingSet = trainingSet
    self.validationSet = validationSet
    self.testingSet = testingSet
    self.detailedSaccadeWidth = detailedSaccadeWidth
    self.detailedSaccadeHeight = detailedSaccadeHeight

    self.net = None
    self.trainingImageIndex = None
    self.networkDutyCycles = None
    self.networkSensor = None
    self.networkSP = None
    self.networkTM = None
    self.networkTP = None
    self.networkClassifier = None
    self.networkSensor = None
    self.numTrainingImages = 0
    self.numTestingImages = 0
    self.trainingImageIndex = 0
    self.testingImageIndex = 0
    self.numCorrect = 0

    if createNetwork:
      self.createNet()


  def createNet(self):
    """ Set up the structure of the network """
    net = Network()

    Network.unregisterRegion(SaccadeSensor.__name__)
    Network.registerRegion(SaccadeSensor)
    Network.unregisterRegion(ExtendedTMRegion.__name__)
    Network.registerRegion(ExtendedTMRegion)
    Network.unregisterRegion(ColumnPoolerRegion.__name__)
    Network.registerRegion(ColumnPoolerRegion)

    imageSensorParams = copy.deepcopy(DEFAULT_IMAGESENSOR_PARAMS)
    if self.loggingDir is not None:
      imageSensorParams["logDir"] = "sensorImages/" + self.loggingDir
      imageSensorParams["logOutputImages"] = 1
      imageSensorParams["logOriginalImages"] = 1
      imageSensorParams["logFilteredImages"] = 1
      imageSensorParams["logLocationImages"] = 1
      imageSensorParams["logLocationOnOriginalImage"] = 1

    net.addRegion("sensor", "py.SaccadeSensor",
                  yaml.dump(imageSensorParams))
    sensor = net.regions["sensor"].getSelf()

    DEFAULT_SP_PARAMS["columnCount"] = sensor.getOutputElementCount("dataOut")
    net.addRegion("SP", "py.SPRegion", yaml.dump(DEFAULT_SP_PARAMS))
    sp = net.regions["SP"].getSelf()

    DEFAULT_TM_PARAMS["columnDimensions"] = (sp.getOutputElementCount("bottomUpOut"),)
    DEFAULT_TM_PARAMS["basalInputWidth"] = sensor.getOutputElementCount("saccadeOut")
    net.addRegion("TM", "py.ExtendedTMRegion", yaml.dump(DEFAULT_TM_PARAMS))

    net.addRegion("TP", "py.ColumnPoolerRegion", yaml.dump(DEFAULT_TP_PARAMS))

    net.addRegion("classifier","py.KNNClassifierRegion",
                  yaml.dump(DEFAULT_CLASSIFIER_PARAMS))


    net.link("sensor", "SP", "UniformLink", "",
             srcOutput="dataOut", destInput="bottomUpIn")
    net.link("SP", "TM", "UniformLink", "",
             srcOutput="bottomUpOut", destInput="activeColumns")
    net.link("sensor", "TM", "UniformLink", "",
             srcOutput="saccadeOut", destInput="basalInput")
    net.link("TM", "TP", "UniformLink", "",
            srcOutput="predictedActiveCells", destInput="feedforwardInput")
    net.link("TP", "TM", "UniformLink", "",
            srcOutput="feedForwardOutput", destInput="apicalInput")
    net.link("TP", "classifier", "UniformLink", "",
             srcOutput="feedForwardOutput", destInput="bottomUpIn")
    #net.link("TM", "classifier", "UniformLink", "",
    #         srcOutput="predictedActiveCells", destInput="bottomUpIn")
    net.link("sensor", "classifier", "UniformLink", "",
             srcOutput="categoryOut", destInput="categoryIn")

    self.net = net
    self.networkSensor = self.net.regions["sensor"]
    self.networkSP = self.net.regions["SP"]
    self.networkTM = self.net.regions["TM"]
    self.networkTP = self.net.regions["TP"]
    self.networkClassifier = self.net.regions["classifier"]


  def loadFromFile(self, filename):
    """ Load a serialized network
    :param filename: Where the network should be loaded from
    """
    print "Loading network from {file}...".format(file=filename)
    Network.unregisterRegion(SaccadeSensor.__name__)
    Network.registerRegion(SaccadeSensor)

    Network.registerRegion(ExtendedTMRegion)

    self.net = Network(filename)

    self.networkSensor = self.net.regions["sensor"]
    self.networkSensor.setParameter("numSaccades", SACCADES_PER_IMAGE_TESTING)

    self.networkSP = self.net.regions["SP"]
    self.networkClassifier = self.net.regions["classifier"]

    self.numCorrect = 0

  def loadExperiment(self):
    """ Load images into ImageSensor and set the learning mode for the SP. """
    print "============= Loading training images ================="
    t1 = time.time()
    self.networkSensor.executeCommand(
        ["loadMultipleImages", self.trainingSet])
    numTrainingImages = self.networkSensor.getParameter("numImages")
    t2 = time.time()
    print "Load time for training images:", t2-t1
    print "Number of training images", numTrainingImages

    self.numTrainingImages = numTrainingImages
    self.trainingImageIndex = 0


  def runNetworkOneImage(self, enableViz=False):
    """ Runs a single image through the network stepping through all saccades

    :param bool enableViz: If true, visualizations are generated and returned
    :return: If enableViz, return a tuple (saccadeImgsList, saccadeDetailList,
      saccadeHistList). saccadeImgsList is a list of images with the fovea
      highlighted. saccadeDetailList is a list of resized images showing the
      contents of the fovea. saccadeHistList shows the fovea history.
      If not enableViz, returns True
      Regardless of enableViz, if False is returned, all images have been
      saccaded over.
    """
    if self.trainingImageIndex < self.numTrainingImages:

      saccadeList = []
      saccadeImgsList = []
      saccadeHistList = []
      saccadeDetailList = []
      originalImage = None

      self.networkTM.executeCommand(["reset"])

      for i in range(SACCADES_PER_IMAGE_TRAINING):
        self.net.run(1)
        if originalImage is None:
          originalImage = deserializeImage(
              yaml.load(self.networkSensor.getParameter("originalImage")))
          imgCenter = (originalImage.size[0] / 2,
                       originalImage.size[1] / 2,)
        saccadeList.append({
            "offset1":
                (yaml.load(
                    self.networkSensor
                    .getParameter("prevSaccadeInfo"))
                 ["prevOffset"]),
            "offset2":
                (yaml.load(
                    self.networkSensor
                    .getParameter("prevSaccadeInfo"))
                 ["newOffset"])})

        if enableViz:
          detailImage = deserializeImage(
              yaml.load(self.networkSensor.getParameter("outputImage")))
          detailImage = detailImage.resize((self.detailedSaccadeWidth,
                                            self.detailedSaccadeHeight),
                                           Image.ANTIALIAS)
          saccadeDetailList.append(ImageTk.PhotoImage(detailImage))

          imgWithSaccade = originalImage.convert("RGB")
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset2"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset2"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] + (_FOVEA_SIZE / 2)),
              fill=(255, 0, 0), width=1) # Left
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset2"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset2"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] + (_FOVEA_SIZE / 2)),
              fill=(255, 0, 0), width=1) # Right
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset2"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset2"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] - (_FOVEA_SIZE / 2)),
              fill=(255, 0, 0), width=1) # Top
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset2"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] + (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset2"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] + (_FOVEA_SIZE / 2)),
              fill=(255, 0, 0), width=1) # Bottom

          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset1"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset1"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] + (_FOVEA_SIZE / 2)),
              fill=(0, 255, 0), width=1) # Left
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset1"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset1"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] + (_FOVEA_SIZE / 2)),
              fill=(0, 255, 0), width=1) # Right
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset1"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset1"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] - (_FOVEA_SIZE / 2)),
              fill=(0, 255, 0), width=1) # Top
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset1"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] + (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset1"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] + (_FOVEA_SIZE / 2)),
              fill=(0, 255, 0), width=1) # Bottom

          saccadeImgsList.append(ImageTk.PhotoImage(imgWithSaccade))

          saccadeHist = originalImage.convert("RGB")
          for i, saccade in enumerate(saccadeList):
            ImageDraw.Draw(saccadeHist).rectangle(
                (imgCenter[0] + saccade["offset2"][0] - _FOVEA_SIZE/2,
                 imgCenter[0] + saccade["offset2"][1] - _FOVEA_SIZE/2,
                 imgCenter[0] + saccade["offset2"][0] + _FOVEA_SIZE/2,
                 imgCenter[0] + saccade["offset2"][1] + _FOVEA_SIZE/2),
                fill=(0,
                      (255/SACCADES_PER_IMAGE_TRAINING*(SACCADES_PER_IMAGE_TRAINING-i)),
                      (255/SACCADES_PER_IMAGE_TRAINING*i)))
          saccadeHist = saccadeHist.resize((self.detailedSaccadeWidth,
                                            self.detailedSaccadeHeight),
                                           Image.ANTIALIAS)
          saccadeHistList.append(ImageTk.PhotoImage(saccadeHist))

      self.trainingImageIndex += 1
      print ("Iteration: {iter}; Category: {cat}"
             .format(iter=self.trainingImageIndex,
                     cat=self.networkSensor.getOutputData("categoryOut")))

      if enableViz:
        return (saccadeImgsList, saccadeDetailList, saccadeHistList,
                self.networkSensor.getOutputData("categoryOut"))
      return True

    else:
      return False


  def runNetworkBatch(self, batchSize):
    """ Run the network in batches.

    :param batchSize: Number of images to show in this batch
    :return: True if there are more images left to be saccaded over.
      Otherwise False.
    """
    startTime = time.time()
    while self.trainingImageIndex < self.numTrainingImages:
      self.networkTM.executeCommand(["reset"])
      for i in range(SACCADES_PER_IMAGE_TRAINING):
        self.net.run(1)

      self.trainingImageIndex += 1
      if self.trainingImageIndex % batchSize == 0:
        print ("Iteration: {iter}; Category: {cat}; Time per batch: {t}"
               .format(iter=self.trainingImageIndex,
                       cat=self.networkSensor.getOutputData("categoryOut"),
                       t=time.time()-startTime))
        return True
    return False

  def setupNetworkTest(self):
    self.networkSensor.executeCommand(["loadMultipleImages", self.testingSet])
    self.numTestingImages = self.networkSensor.getParameter("numImages")
    self.testingImageIndex = 0

    print "NumTestingImages {test}".format(test=self.numTestingImages)


  def testNetworkOneImage(self, enableViz=False):
    if self.testingImageIndex < self.numTestingImages:
      saccadeList = []
      saccadeImgsList = []
      saccadeHistList = []
      saccadeDetailList = []
      inferredCategoryList = []
      originalImage = None

      self.networkTM.executeCommand(["reset"])
      for i in range(SACCADES_PER_IMAGE_TESTING):
        self.net.run(1)
        if originalImage is None:
          originalImage = deserializeImage(
              yaml.load(self.networkSensor.getParameter("originalImage")))
          imgCenter = (originalImage.size[0] / 2,
                       originalImage.size[1] / 2,)
        saccadeList.append({
            "offset1":
                (yaml.load(
                    self.networkSensor
                    .getParameter("prevSaccadeInfo"))
                 ["prevOffset"]),
            "offset2":
                (yaml.load(
                    self.networkSensor
                    .getParameter("prevSaccadeInfo"))
                 ["newOffset"])})
        inferredCategoryList.append(
            self.networkClassifier.getOutputData("categoriesOut").argmax())

        if enableViz:
          detailImage = deserializeImage(
              yaml.load(self.networkSensor.getParameter("outputImage")))
          detailImage = detailImage.resize((self.detailedSaccadeWidth,
                                            self.detailedSaccadeHeight),
                                           Image.ANTIALIAS)
          saccadeDetailList.append(ImageTk.PhotoImage(detailImage))

          imgWithSaccade = originalImage.convert("RGB")
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset2"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset2"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] + (_FOVEA_SIZE / 2)),
              fill=(255, 0, 0), width=1) # Left
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset2"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset2"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] + (_FOVEA_SIZE / 2)),
              fill=(255, 0, 0), width=1) # Right
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset2"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset2"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] - (_FOVEA_SIZE / 2)),
              fill=(255, 0, 0), width=1) # Top
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset2"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] + (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset2"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset2"][1] + (_FOVEA_SIZE / 2)),
              fill=(255, 0, 0), width=1) # Bottom

          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset1"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset1"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] + (_FOVEA_SIZE / 2)),
              fill=(0, 255, 0), width=1) # Left
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset1"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset1"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] + (_FOVEA_SIZE / 2)),
              fill=(0, 255, 0), width=1) # Right
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset1"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] - (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset1"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] - (_FOVEA_SIZE / 2)),
              fill=(0, 255, 0), width=1) # Top
          ImageDraw.Draw(imgWithSaccade).line(
              (imgCenter[0] + saccadeList[i]["offset1"][0] + (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] + (_FOVEA_SIZE / 2),
               imgCenter[0] + saccadeList[i]["offset1"][0] - (_FOVEA_SIZE / 2),
               imgCenter[1] + saccadeList[i]["offset1"][1] + (_FOVEA_SIZE / 2)),
              fill=(0, 255, 0), width=1) # Bottom

          saccadeImgsList.append(ImageTk.PhotoImage(imgWithSaccade))

          saccadeHist = originalImage.convert("RGB")
          for i, saccade in enumerate(saccadeList):
            ImageDraw.Draw(saccadeHist).rectangle(
                (imgCenter[0] + saccade["offset2"][0] - _FOVEA_SIZE/2,
                 imgCenter[0] + saccade["offset2"][1] - _FOVEA_SIZE/2,
                 imgCenter[0] + saccade["offset2"][0] + _FOVEA_SIZE/2,
                 imgCenter[0] + saccade["offset2"][1] + _FOVEA_SIZE/2),
                fill=(0,
                      (255/SACCADES_PER_IMAGE_TESTING*(SACCADES_PER_IMAGE_TESTING-i)),
                      (255/SACCADES_PER_IMAGE_TESTING*i)))
          saccadeHist = saccadeHist.resize((self.detailedSaccadeWidth,
                                            self.detailedSaccadeHeight),
                                           Image.ANTIALIAS)
          saccadeHistList.append(ImageTk.PhotoImage(saccadeHist))

      inferredCategory = self._getMostCommonCategory(inferredCategoryList)
      isCorrectClassification = False
      if self.networkSensor.getOutputData("categoryOut") == inferredCategory:
        isCorrectClassification = True
        self.numCorrect += 1
      print ("Iteration: {iter}; Category: {cat}"
             .format(iter=self.testingImageIndex,
                     cat=self.networkSensor.getOutputData("categoryOut")))
      self.testingImageIndex += 1

      if enableViz:
        return (saccadeImgsList, saccadeDetailList, saccadeHistList,
                inferredCategoryList,
                self.networkSensor.getOutputData("categoryOut"),
                isCorrectClassification)
      return (True, isCorrectClassification)

    else:
      return False


  def testNetworkBatch(self, batchSize):
    if self.testingImageIndex >= self.numTestingImages:
      return False

    while self.testingImageIndex < self.numTestingImages:
      inferredCategoryList = []
      self.networkTM.executeCommand(["reset"])
      for i in range(SACCADES_PER_IMAGE_TESTING):
        self.net.run(1)
        inferredCategoryList.append(
            self.networkClassifier.getOutputData("categoriesOut").argmax())
      inferredCategory = self._getMostCommonCategory(inferredCategoryList)
      if self.networkSensor.getOutputData("categoryOut") == inferredCategory:
        self.numCorrect += 1

      self.testingImageIndex += 1

      if self.testingImageIndex % batchSize == 0:
        print ("Testing iteration: {iter}"
               .format(iter=self.testingImageIndex))
        break

    return self.numCorrect


  @staticmethod
  def _getMostCommonCategory(categoryList):
    return collections.Counter(categoryList).most_common(1)[0][0]


  def setLearningMode(self,
                      learningSP=False,
                      learningTM=False,
                      learningTP=False,
                      learningClassifier=False):
    if learningSP:
      self.networkSP.setParameter("learningMode", 1)
      self.networkSP.setParameter("inferenceMode", 0)
    else:
      self.networkSP.setParameter("learningMode", 0)
      self.networkSP.setParameter("inferenceMode", 1)

    if learningTM:
      self.networkTM.setParameter("learn", 1)
    else:
      self.networkTM.setParameter("learn", 0)

    if learningTM:
      self.networkTP.setParameter("learningMode", 1)
    else:
      self.networkTP.setParameter("learningMode", 0)

    if learningClassifier:
      self.networkClassifier.setParameter("learningMode", 1)
      self.networkClassifier.setParameter("inferenceMode", 0)
    else:
      self.networkClassifier.setParameter("learningMode", 0)
      self.networkClassifier.setParameter("inferenceMode", 1)


  def saveNetwork(self):
    print "Saving network at {path}".format(path=self.netFile)
    self.net.save(self.netFile)


  def resetIndex(self):
    self.trainingImageIndex = 0
