#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import copy
import time

import numpy
from PIL import ImageTk, ImageDraw, Image
import yaml

from nupic.bindings.math import GetNTAReal
from nupic.engine import Network

from nupicvision.regions.ImageSensor import ImageSensor
from nupicvision.image import deserializeImage



SACCADES_PER_IMAGE = 20
_SACCADE_SIZE = 5
_FOVEA_SIZE = 10
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
        "numSaccades": SACCADES_PER_IMAGE,
        "maxDrift": 0,
        "seed": 1738}]),
    "postFilters": yaml.dump([["Resize", {
        "size": (_FOVEA_SIZE, _FOVEA_SIZE), "method": "center" }]]),
}

DEFAULT_SP_PARAMS = {
    "columnCount": 4096,
    "spatialImp": "cpp",
    "inputWidth": 1024,
    "spVerbosity": 1,
    "synPermConnected": 0.2,
    "synPermActiveInc": 0.0,
    "synPermInactiveDec": 0.0,
    "seed": 1956,
    "numActiveColumnsPerInhArea": 240,
    "globalInhibition": 1,
    "potentialPct": 0.9,
    "maxBoost": 1.0
}

DEFAULT_CLASSIFIER_PARAMS = {
    "distThreshold": 0.000001,
    "maxCategoryCount": 10,
}



class SaccadeNetwork(object):
  """
  A HTM network structured as follows:
  ImageSensor (RandomSaccade) -> SP -> ?
  """
  def __init__(self,
               networkName,
               trainingSet,
               testingSet,
               loggingDir = None,
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
      return from the runNetworkOneImage
    :param int detailedSaccadeHeight: (optional) Height of detailed saccades to
      return from the runNetworkOneImage
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
    self.networkClassifier = None
    self.networkDutyCycles = None
    self.networkSP = None
    self.networkSensor = None
    self.numTrainingImages = 0
    self.networkPySP = None

    if createNetwork:
      self.createNet()

  def createNet(self):
    """ Set up the structure of the network """
    net = Network()

    Network.unregisterRegion(ImageSensor.__name__)
    Network.registerRegion(ImageSensor)

    imageSensorParams = copy.deepcopy(DEFAULT_IMAGESENSOR_PARAMS)
    if self.loggingDir is not None:
      imageSensorParams["logDir"] = "sensorImages/" + self.loggingDir
      imageSensorParams["logOutputImages"] = 1
      imageSensorParams["logOriginalImages"] = 1
      imageSensorParams["logFilteredImages"] = 1
      imageSensorParams["logLocationImages"] = 1
      imageSensorParams["logLocationOnOriginalImage"] = 1

    net.addRegion("sensor", "py.ImageSensor",
                  yaml.dump(imageSensorParams))
    net.addRegion("SP", "py.SPRegion", yaml.dump(DEFAULT_SP_PARAMS))
    net.addRegion("classifier","py.KNNClassifierRegion",
                  yaml.dump(DEFAULT_CLASSIFIER_PARAMS))

    net.link("sensor", "SP", "UniformLink", "",
             srcOutput = "dataOut", destInput = "bottomUpIn")
    net.link("SP", "classifier", "UniformLink", "",
             srcOutput = "bottomUpOut", destInput = "bottomUpIn")
    net.link("sensor", "classifier", "UniformLink", "",
             srcOutput = "categoryOut", destInput = "categoryIn")

    self.net = net


  def loadExperiment(self):
    """ Load images into ImageSensor and set the learning mode for the SP. """
    self.networkSensor = self.net.regions["sensor"]
    self.networkSP = self.net.regions["SP"]
    self.networkPySP = self.networkSP.getSelf()
    self.networkClassifier = self.net.regions["classifier"]
    self.networkDutyCycles = numpy.zeros(DEFAULT_SP_PARAMS["columnCount"],
                                         dtype=GetNTAReal())

    print "============= Loading training images ================="
    t1 = time.time()
    self.networkSensor.executeCommand(
        ["loadMultipleImages", self.trainingSet])
    numTrainingImages = self.networkSensor.getParameter("numImages")
    t2 = time.time()
    print "Load time for training images:", t2-t1
    print "Number of training images", numTrainingImages

    # Set up the SP parameters
    print "============= SP training ================="
    self.networkClassifier.setParameter("inferenceMode", 0)
    self.networkClassifier.setParameter("learningMode", 0)
    self.networkSP.setParameter("learningMode", 1)
    self.networkSP.setParameter("inferenceMode", 0)
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
      for i in range(SACCADES_PER_IMAGE):
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
                      (255/SACCADES_PER_IMAGE*(SACCADES_PER_IMAGE-i)),
                      (255/SACCADES_PER_IMAGE*i)))
          saccadeHist = saccadeHist.resize((self.detailedSaccadeWidth,
                                            self.detailedSaccadeHeight),
                                           Image.ANTIALIAS)
          saccadeHistList.append(ImageTk.PhotoImage(saccadeHist))
        self.networkDutyCycles += self.networkPySP._spatialPoolerOutput #pylint: disable=W0212

      print ("Iteration: {iter}; Category: {cat}"
             .format(iter=self.trainingImageIndex,
                     cat=self.networkSensor.getOutputData("categoryOut")))
      self.trainingImageIndex += 1

      if enableViz:
        return (saccadeImgsList, saccadeDetailList, saccadeHistList)
      return True

    else:
      return False


  def run(self):
    """ Run the network until all images have been seen """
    while self.trainingImageIndex < self.numTrainingImages:
      for i in range(SACCADES_PER_IMAGE):
        self.net.run(1)

        self.networkDutyCycles += self.networkPySP._spatialPoolerOutput #pylint: disable=W0212

      if self.trainingImageIndex % (self.numTrainingImages/100) == 0:
        print ("Iteration: {iter}; Category: {cat}"
               .format(iter=self.trainingImageIndex,
                       cat=self.networkSensor.getOutputData("categoryOut")))
      self.trainingImageIndex += 1

    print "Done training SP!"


  def runNetworkBatch(self, batchSize):
    """ Run the network in batches.

    :param batchSize: Number of images to show in this batch
    :return: True if there are more images left to be saccaded over.
      Otherwise False.
    """
    while self.trainingImageIndex < self.numTrainingImages:
      for i in range(SACCADES_PER_IMAGE):
        self.net.run(1)

        self.networkDutyCycles += self.networkPySP._spatialPoolerOutput #pylint: disable=W0212

      if self.trainingImageIndex % batchSize == 0:
        print ("Iteration: {iter}; Category: {cat}"
               .format(iter=self.trainingImageIndex,
                       cat=self.networkSensor.getOutputData("categoryOut")))

      self.trainingImageIndex += 1
      if self.trainingImageIndex % batchSize == 1:
        return True

    print "Done training SP!"
    return False
