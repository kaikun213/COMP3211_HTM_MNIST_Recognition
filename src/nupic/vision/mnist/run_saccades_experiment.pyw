#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

import datetime
import os
import Queue
import threading
import Tkinter as Tk
import tkMessageBox
import ttk as Ttk

from enum import Enum

from nupic.vision.mnist.saccade_network import (
    SaccadeNetwork,
    SACCADES_PER_IMAGE_TRAINING,
    SACCADES_PER_IMAGE_TESTING,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)



# GUI related variables
_APP_WIDTH = 900
_APP_HEIGHT = 400

_APP_SACCADE_FRAME_HEIGHT = _APP_HEIGHT - 50
_APP_SACCADE_LIST_FRAME_WIDTH = _APP_WIDTH * 2 / 3 - 10
_APP_SACCADE_DETAIL_FRAME_WIDTH = _APP_WIDTH * 1 / 3 - 10
_APP_DETAILED_SACCADE_WIDTH = 100
_APP_DETAILED_SACCADE_HEIGHT = 100

# Where to save serialized networks
_NETWORK_DIR_NAME = "networks"



class NetworkMode(Enum):
  TRAINING_MODE = 1
  TESTING_MODE = 2



class MainGUI(object):
  """ A GUI for running the saccade experiment. Allows the user to run the
  experiment with visualizations of individual saccades and use this to tune
  the parameters for making saccades.
  """


  def __init__(self, root,
               loggingDir,
               networkName,
               trainingSet="mnist/training",
               validationSet=None,
               testingSet="mnist/testing"):
    """
    :param root: TK window to build the GUI in
    :param loggingDir: Directory to store logging images
    :param networkName: Used for serializing the network
    :param trainingSet: Directory of training images
    :param validationSet: Directory of validation images (not implemented yet)
    :param testingSet: Directory of testing images
    """
    self.root = root

    # Split main into upper + lower (saccade viewer + control section)
    self.saccadeFrame = Tk.Frame(self.root)
    self.controlFrame = Tk.Frame(self.root)
    self.controlFrame.pack(side=Tk.BOTTOM, fill=Tk.X,)
    self.saccadeFrame.pack(side=Tk.TOP, fill=Tk.X)

    # Split saccade viewer into two sections (list view + detail view)
    self.saccadeListFrame = Tk.Frame(self.saccadeFrame)
    self.saccadeDetailFrame = Tk.Frame(self.saccadeFrame)
    self.saccadeDetailFrame.pack(side=Tk.RIGHT,
                                 fill=Tk.Y,)
    self.saccadeListFrame.pack(side=Tk.LEFT,
                               fill=Tk.Y,)

    # Add images (labels) to saccade list frame
    self.saccadeImgs = []
    maxImgsPerRow = _APP_SACCADE_LIST_FRAME_WIDTH / (IMAGE_WIDTH + 4)
    for i in range(SACCADES_PER_IMAGE_TRAINING):
      self.saccadeImgs.append(Tk.Label(self.saccadeListFrame,
                                       height=IMAGE_HEIGHT,
                                       width=IMAGE_WIDTH))
      self.saccadeImgs[i].image = None
      self.saccadeImgs[i].grid(row=(i / maxImgsPerRow),
                               column=(i % maxImgsPerRow))

    # Split saccade detail frame into image and control sections
    self.saccadeDetailImagesFrame = Tk.Frame(self.saccadeDetailFrame)
    self.saccadeDetailTransportFrame = Tk.Frame(self.saccadeDetailFrame)
    self.saccadeDetailTransportFrame.pack(side=Tk.BOTTOM,
                                          fill=Tk.X) #may not need fill X
    self.saccadeDetailImagesFrame.pack(side=Tk.TOP,
                                       fill=Tk.X) #possiby only need y

    # Add controls to transport frame
    self.buttonNextImg = Tk.Button(self.saccadeDetailTransportFrame,
                                   text="Next image",
                                   command=self.buttonNextImgCb,
                                   state=Tk.DISABLED)
    self.buttonScrubFirst = Tk.Button(self.saccadeDetailTransportFrame,
                                      text="<<",
                                      command=self.buttonScrubFirstCb,
                                      state=Tk.DISABLED)
    self.buttonScrubRev = Tk.Button(self.saccadeDetailTransportFrame,
                                    text="<",
                                    command=self.buttonScrubRevCb,
                                    state=Tk.DISABLED)
    self.buttonScrubFwd = Tk.Button(self.saccadeDetailTransportFrame,
                                    text=">",
                                    command=self.buttonScrubFwdCb,
                                    state=Tk.DISABLED)
    self.buttonScrubLast = Tk.Button(self.saccadeDetailTransportFrame,
                                     text=">>",
                                     command=self.buttonScrubLastCb,
                                     state=Tk.DISABLED)
    self.buttonScrubLast.pack(side=Tk.RIGHT)
    self.buttonScrubFwd.pack(side=Tk.RIGHT)
    self.buttonScrubRev.pack(side=Tk.RIGHT)
    self.buttonScrubFirst.pack(side=Tk.RIGHT)
    self.buttonNextImg.pack(side=Tk.RIGHT)


    # Add images (Tk.Label's) to saccade detail frame
    self.saccadeHistImgList = [None for i in range(SACCADES_PER_IMAGE_TRAINING)]
    self.saccadeHistImg = Tk.Label(self.saccadeDetailImagesFrame,
                                   height=_APP_DETAILED_SACCADE_HEIGHT,
                                   width=_APP_DETAILED_SACCADE_WIDTH)
    self.saccadeHistImg.image = None
    self.saccadeHistLabel = Tk.Label(self.saccadeDetailImagesFrame,
                                     text="History")
    self.saccadeDetailImgList = [None for i in range(SACCADES_PER_IMAGE_TRAINING)]
    self.saccadeCurImage = Tk.Label(self.saccadeDetailImagesFrame,
                                    height=_APP_DETAILED_SACCADE_HEIGHT,
                                    width=_APP_DETAILED_SACCADE_WIDTH)
    self.saccadeCurImage.image = None
    self.saccadeCurLabel = Tk.Label(self.saccadeDetailImagesFrame,
                                    text="Current saccade")
    self.saccadeCategoryList = [None for i in range(SACCADES_PER_IMAGE_TRAINING)]
    self.saccadeCategoryVar = Tk.StringVar()
    self.saccadeCategoryVar.set("Category: ")
    self.saccadeCategoryLabel = Tk.Label(self.saccadeDetailFrame,
                                         textvariable=self.saccadeCategoryVar)

    self.saccadeHistImg.pack(side=Tk.TOP, expand=Tk.YES, fill=Tk.X)
    self.saccadeHistLabel.pack(side=Tk.TOP, fill=Tk.X)
    self.saccadeCurImage.pack(side=Tk.TOP, expand=Tk.YES, fill=Tk.X)
    self.saccadeCurLabel.pack(side=Tk.TOP, fill=Tk.X)
    self.saccadeCategoryLabel.pack(side=Tk.TOP, fill=Tk.X)


    # Control section
    self.controlButtonsFrame = Tk.Frame(self.controlFrame)
    self.controlProgBarFrame = Tk.Frame(self.controlFrame)
    self.controlButtonsFrame.pack(side=Tk.TOP, fill=Tk.BOTH)
    self.controlProgBarFrame.pack(side=Tk.BOTTOM, fill=Tk.BOTH)

    self.controlProgBarLearningFrame = Tk.Frame(self.controlProgBarFrame)
    self.controlProgBarRunningFrame = Tk.Frame(self.controlProgBarFrame)
    self.controlProgBarLearningFrame.pack(side=Tk.LEFT, fill=Tk.BOTH)
    self.controlProgBarRunningFrame.pack(side=Tk.RIGHT, fill=Tk.BOTH)

    # Add buttons to control section
    self.buttonLoadTraining = Tk.Button(self.controlButtonsFrame,
                                        text="Load (train)...",
                                        command=self.buttonLoadTrainingCb)
    self.buttonLoadTesting = Tk.Button(self.controlButtonsFrame,
                                       text="Load (test)...",
                                       command=self.buttonLoadTestingCb)
    self.buttonRunTraining = Tk.Button(self.controlButtonsFrame,
                                       text="Train",
                                       command=self.buttonRunTrainingCb,
                                       state=Tk.DISABLED)
    self.buttonRunTesting = Tk.Button(self.controlButtonsFrame,
                                      text="Test",
                                      command=self.buttonRunTestingCb,
                                      state=Tk.DISABLED)

    self.enableVisualizations = Tk.IntVar()
    self.enableVisualizationsCheckbox = Tk.Checkbutton(
        self.controlButtonsFrame, text="Show visualizations",
        variable=self.enableVisualizations)
    self.enableVisualizationsCheckbox.select()
    self.buttonLoadTraining.pack(side=Tk.LEFT)
    self.buttonLoadTesting.pack(side=Tk.LEFT)
    self.buttonRunTraining.pack(side=Tk.LEFT)
    self.buttonRunTesting.pack(side=Tk.LEFT)
    self.enableVisualizationsCheckbox.pack(side=Tk.LEFT)

    # Add progress bars
    self.progressbarLearning = Ttk.Progressbar(self.controlProgBarLearningFrame,
                                               orient=Tk.HORIZONTAL,
                                               length=_APP_WIDTH*3/4-5)
    self.progressbarRunning = Ttk.Progressbar(self.controlProgBarRunningFrame,
                                              orient=Tk.HORIZONTAL,
                                              length=_APP_WIDTH*1/4-5)
    self.progressbarLearning.pack()
    self.progressbarRunning.pack()


    # Other setup
    self.currentDetailSaccadeIndex = 0

    # Set up the networks & queue
    self.eventQueue = Queue.Queue()
    self.networkName = networkName
    self.trainingNetwork = SaccadeNetwork(
        loggingDir=loggingDir, networkName=networkName,
        trainingSet=trainingSet, validationSet=validationSet,
        testingSet=testingSet,
        detailedSaccadeWidth=_APP_DETAILED_SACCADE_WIDTH,
        detailedSaccadeHeight=_APP_DETAILED_SACCADE_HEIGHT,
        createNetwork=False)
    self.testingNetwork = SaccadeNetwork(
        loggingDir=loggingDir, networkName=networkName,
        trainingSet=trainingSet, validationSet=validationSet,
        testingSet=testingSet,
        detailedSaccadeWidth=_APP_DETAILED_SACCADE_WIDTH,
        detailedSaccadeHeight=_APP_DETAILED_SACCADE_HEIGHT,
        createNetwork=False)
    self.networkMode = None
    # Special variables used for testing
    self.currentTestImgIsCorrect = None
    self.currentTestImgActualCategory = None


  def buttonLoadTrainingCb(self):
    """ Callback for the "Load" button. Creates a network and loads
    the training images for the experiment.
    """
    self.networkMode = NetworkMode.TRAINING_MODE
    self.trainingNetwork.createNet()
    self.trainingNetwork.loadExperiment()
    self.trainingNetwork.setLearningMode(learningSP=True,
                                         learningTM=False,
                                         learningTP=False,
                                         learningClassifier=False)

    # Update GUI
    self.buttonLoadTraining.config(state=Tk.DISABLED)
    self.buttonLoadTesting.config(state=Tk.DISABLED)
    self.buttonRunTraining.config(state=Tk.NORMAL)
    self.buttonNextImg.config(state=Tk.NORMAL)
    self.buttonNextImgCb()


  def buttonNextImgCb(self):
    """ Callback for the "Next" button. Advances the network one training
    image.
    """
    if self.networkMode == NetworkMode.TRAINING_MODE:
      if self.enableVisualizations.get():
        result = self.trainingNetwork.runNetworkOneImage(enableViz=True)

        if result == False:
          # The sp has no more images to learn
          self.buttonNextImg.config(state=Tk.DISABLED)
          self.buttonRunTraining.config(state=Tk.DISABLED)
          tkMessageBox.showinfo("INFO", "SP learning is done!")
          return

        (saccadeImgs, saccadeDetailImgs, saccadeHistImgs, categoryOut) = result
        for i in range(SACCADES_PER_IMAGE_TRAINING):
          self.saccadeImgs[i].configure(image=saccadeImgs[i])
          self.saccadeImgs[i].image = saccadeImgs[i]
          self.saccadeImgs[i].configure(bg="white")

          self.saccadeHistImgList[i] = saccadeHistImgs[i]
          self.saccadeDetailImgList[i] = saccadeDetailImgs[i]

        self.currentDetailSaccadeIndex = SACCADES_PER_IMAGE_TRAINING-1 # Last saccade
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="red")
        self.saccadeCategoryVar.set("Category = {cat}".format(cat=categoryOut))

        self.buttonScrubFirst.config(state=Tk.NORMAL)
        self.buttonScrubRev.config(state=Tk.NORMAL)
        self.buttonScrubLast.config(state=Tk.DISABLED)
        self.buttonScrubFwd.config(state=Tk.DISABLED)
        self.updateDetailImages()

      else:
        result = self.trainingNetwork.runNetworkOneImage(enableViz=False)

        if result == False:
          self.buttonNextImg.config(state=Tk.DISABLED)
          self.buttonRunTraining.config(state=Tk.DISABLED)
          tkMessageBox.showinfo("INFO", "SP learning is done!")
          return
    elif self.networkMode == NetworkMode.TESTING_MODE:
      if self.enableVisualizations.get():
        result = self.testingNetwork.testNetworkOneImage(enableViz=True)

        if result == False:
          # The sp has no more images to learn
          self.buttonNextImg.config(state=Tk.DISABLED)
          self.buttonRunTesting.config(state=Tk.DISABLED)
          tkMessageBox.showinfo("INFO",
                                "Testing is done! numCorrect = {num}"
                                .format(num=self.testingNetwork.numCorrect))
          return

        (saccadeImgs, saccadeDetailImgs, saccadeHistImgs,
         inferredCategoryList, currentTestImgActualCategory,
         currentTestImgIsCorrect) = result
        for i in range(SACCADES_PER_IMAGE_TESTING):
          self.saccadeImgs[i].configure(image=saccadeImgs[i])
          self.saccadeImgs[i].image = saccadeImgs[i]
          if inferredCategoryList[i] == currentTestImgActualCategory:
            self.saccadeImgs[i].configure(bg="green")
          else:
            self.saccadeImgs[i].configure(bg="red")

          self.saccadeHistImgList[i] = saccadeHistImgs[i]
          self.saccadeDetailImgList[i] = saccadeDetailImgs[i]
          self.saccadeCategoryList[i] = inferredCategoryList[i]
        self.currentTestImgIsCorrect = currentTestImgIsCorrect
        self.currentTestImgActualCategory = currentTestImgActualCategory

        self.currentDetailSaccadeIndex = SACCADES_PER_IMAGE_TESTING-1 # Last saccade
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="blue")

        self.buttonScrubFirst.config(state=Tk.NORMAL)
        self.buttonScrubRev.config(state=Tk.NORMAL)
        self.buttonScrubLast.config(state=Tk.DISABLED)
        self.buttonScrubFwd.config(state=Tk.DISABLED)
        self.updateDetailImages()

      else:
        result = self.testingNetwork.testNetworkOneImage(enableViz=False)

        if result == False:
          self.buttonNextImg.config(state=Tk.DISABLED)
          self.buttonRunTesting.config(state=Tk.DISABLED)
          tkMessageBox.showinfo("INFO",
                                "Testing is done! numCorrect = {num}"
                                .format(num=self.testingNetwork.numCorrect))
          return



  def buttonScrubFirstCb(self):
    """ Callback for the "First" transport button. Go to the first saccade
    into the detailed saccade section """
    if self.networkMode == NetworkMode.TESTING_MODE:
      if (self.saccadeCategoryList[self.currentDetailSaccadeIndex] ==
          self.currentTestImgActualCategory):
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="green")
      else:
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="red")
    else:
      self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="white")
    self.currentDetailSaccadeIndex = 0
    self.buttonScrubFirst.config(state=Tk.DISABLED)
    self.buttonScrubRev.config(state=Tk.DISABLED)
    self.buttonScrubLast.config(state=Tk.NORMAL)
    self.buttonScrubFwd.config(state=Tk.NORMAL)
    self.updateDetailImages()


  def buttonScrubRevCb(self):
    """ Callback for the "Reverse" transport button. Go back by one saccade
    in the detailed saccade section """
    if self.networkMode == NetworkMode.TESTING_MODE:
      if (self.saccadeCategoryList[self.currentDetailSaccadeIndex] ==
          self.currentTestImgActualCategory):
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="green")
      else:
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="red")
    else:
      self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="white")
    self.currentDetailSaccadeIndex -=1
    if self.currentDetailSaccadeIndex == 0:
      self.buttonScrubFirst.config(state=Tk.DISABLED)
      self.buttonScrubRev.config(state=Tk.DISABLED)
    self.buttonScrubLast.config(state=Tk.NORMAL)
    self.buttonScrubFwd.config(state=Tk.NORMAL)
    self.updateDetailImages()


  def buttonScrubFwdCb(self):
    """ Callback for the "Forward" transport button. Go forward by one saccade
    in the detailed saccade section """
    if self.networkMode == NetworkMode.TESTING_MODE:
      if (self.saccadeCategoryList[self.currentDetailSaccadeIndex] ==
          self.currentTestImgActualCategory):
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="green")
      else:
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="red")
    else:
      self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="white")
    self.currentDetailSaccadeIndex +=1


    if self.currentDetailSaccadeIndex == (SACCADES_PER_IMAGE_TRAINING
                                          if self.networkMode == NetworkMode.TRAINING_MODE
                                          else SACCADES_PER_IMAGE_TESTING)-1:
      self.buttonScrubLast.config(state=Tk.DISABLED)
      self.buttonScrubFwd.config(state=Tk.DISABLED)
    self.buttonScrubFirst.config(state=Tk.NORMAL)
    self.buttonScrubRev.config(state=Tk.NORMAL)
    self.updateDetailImages()


  def buttonScrubLastCb(self):
    """ Callback for the "Last" transport button. Go to the last saccade
    in the detailed saccade section """
    if self.networkMode == NetworkMode.TESTING_MODE:
      if (self.saccadeCategoryList[self.currentDetailSaccadeIndex] ==
          self.currentTestImgActualCategory):
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="green")
      else:
        self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="red")
    else:
      self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="white")
    self.currentDetailSaccadeIndex = (SACCADES_PER_IMAGE_TRAINING
                                      if self.networkMode == NetworkMode.TRAINING_MODE
                                      else SACCADES_PER_IMAGE_TESTING)-1
    self.buttonScrubLast.config(state=Tk.DISABLED)
    self.buttonScrubFwd.config(state=Tk.DISABLED)
    self.buttonScrubFirst.config(state=Tk.NORMAL)
    self.buttonScrubRev.config(state=Tk.NORMAL)
    self.updateDetailImages()


  def updateDetailImages(self):
    """ Update the detailed saccade view with the current highlighted saccade
    (called by the various transport buttons and the "Next" button)
    """
    self.saccadeHistImg.configure(
        image=self.saccadeHistImgList[self.currentDetailSaccadeIndex])
    self.saccadeHistImg.image = self.saccadeHistImgList[
        self.currentDetailSaccadeIndex]
    self.saccadeCurImage.configure(
        image=self.saccadeDetailImgList[self.currentDetailSaccadeIndex])
    self.saccadeCurImage.image = self.saccadeDetailImgList[
        self.currentDetailSaccadeIndex]
    self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="blue")

    if self.networkMode == NetworkMode.TESTING_MODE:
      if self.currentTestImgIsCorrect:
        self.saccadeCategoryVar.set(
            "Inference Worked! Actual={actual}; Inferred={inferred}"
            .format(actual=self.currentTestImgActualCategory,
                    inferred=self.saccadeCategoryList[
                        self.currentDetailSaccadeIndex]))
      else:
        self.saccadeCategoryVar.set(
            "Inference Failed! Actual={actual}; Inferred={inferred}".format(
                actual=self.currentTestImgActualCategory,
                inferred=self.saccadeCategoryList[
                    self.currentDetailSaccadeIndex]))


  def buttonRunTrainingCb(self):
    """ Runs the training on the network.
    NOTE: Automatically disables visualizations for speed
    """
    self.enableVisualizationsCheckbox.deselect()
    self.buttonNextImg.config(state=Tk.DISABLED)
    self.buttonRunTraining.config(state=Tk.DISABLED)
    self.buttonScrubLast.config(state=Tk.DISABLED)
    self.buttonScrubFwd.config(state=Tk.DISABLED)
    self.buttonScrubFirst.config(state=Tk.DISABLED)
    self.buttonScrubRev.config(state=Tk.DISABLED)

    threading.Thread(target=self.runNetworkBatch,
                     kwargs={"network": self.trainingNetwork,
                             "queue": self.eventQueue,
                             "process": "SP"}).start()

    self.root.after(100, self.processEventQueue)


  @staticmethod
  def runNetworkBatch(network, queue, process=""):
    """ A static function that allows the network to be run in batches from
    threads that won't block the GUI from updating.

    :param network: An initialized network
    :param queue: The queue to send results to
    :param process: The name of the process. Used in sending results to queue
    """
    result = network.runNetworkBatch(batchSize=1)
    queue.put({"process": process,
               "running": result})


  def buttonLoadTestingCb(self):
    self.networkMode = NetworkMode.TESTING_MODE
    #self.networkName = "{networksDir}/{network}".format(
    #    networksDir=_NETWORK_DIR_NAME,
    #    network=os.listdir(_NETWORK_DIR_NAME)[-1])
    #self.testingNetwork.loadFromFile(self.networkName)
    self.testingNetwork = self.trainingNetwork
    self.testingNetwork.networkSensor.setParameter(
      "numSaccades", SACCADES_PER_IMAGE_TESTING)


    self.testingNetwork.setLearningMode(learningSP=False,
                                        learningTM=False,
                                        learningTP=True,
                                        learningClassifier=True)

    print "Loading testing images..."
    self.testingNetwork.setupNetworkTest()

    # GUI Setup
    self.buttonRunTesting.config(state=Tk.NORMAL)
    self.buttonLoadTraining.config(state=Tk.DISABLED)
    self.buttonLoadTesting.config(state=Tk.DISABLED)
    self.buttonNextImg.config(state=Tk.NORMAL)
    self.enableVisualizationsCheckbox.select()
    self.buttonNextImgCb()

    if SACCADES_PER_IMAGE_TESTING < SACCADES_PER_IMAGE_TRAINING:
      for i in range(SACCADES_PER_IMAGE_TESTING, SACCADES_PER_IMAGE_TRAINING):
        self.saccadeImgs[i].destroy()



  def buttonRunTestingCb(self):
    self.buttonRunTesting.config(state=Tk.DISABLED)
    self.enableVisualizationsCheckbox.deselect()
    self.buttonNextImg.config(state=Tk.DISABLED)
    self.buttonScrubLast.config(state=Tk.DISABLED)
    self.buttonScrubFwd.config(state=Tk.DISABLED)
    self.buttonScrubFirst.config(state=Tk.DISABLED)
    self.buttonScrubRev.config(state=Tk.DISABLED)

    print "Running test..."
    threading.Thread(target=self.testNetworkBatch,
                     kwargs={"network": self.testingNetwork,
                             "queue": self.eventQueue,
                             "process": "TEST"}).start()
    self.root.after(100, self.processEventQueue)


  @staticmethod
  def testNetworkBatch(network, queue, process=""):
    result = network.testNetworkBatch(batchSize=1)
    if result is not False:
      queue.put({"process": process,
                 "running": True})
    else:
      queue.put({"process": process,
                 "running": False})


  def processEventQueue(self):
    """ GUI method periodically called when running large jobs on the network.
    Updates progress bars with network training/testing progress.
    """
    try:
      msg = self.eventQueue.get(0)

      # SP
      if (msg["process"] == "SP" and
          msg["running"] == False):
        print "SP learning is done!"
        self.progressbarRunning.stop()
        self.progressbarLearning.stop()
        # TM
        self.trainingNetwork.setLearningMode(learningSP=False,
                                             learningTM=True,
                                             learningTP=False,
                                             learningClassifier=False)
        self.trainingNetwork.resetIndex()
        threading.Thread(target=self.runNetworkBatch,
                         kwargs={"network": self.trainingNetwork,
                                 "queue": self.eventQueue,
                                 "process": "TM"}).start()
        self.root.after(100, self.processEventQueue)
      elif (msg["process"] == "SP" and
            msg["running"] == True):
        self.progressbarLearning.step()
        threading.Thread(target=self.runNetworkBatch,
                         kwargs={"network": self.trainingNetwork,
                                 "queue": self.eventQueue,
                                 "process": "SP"}).start()
        self.root.after(100, self.processEventQueue)

      # TM
      elif (msg["process"] == "TM" and
          msg["running"] == False):
        print "TM learning is done!"
        self.progressbarRunning.stop()
        self.progressbarLearning.stop()
        # Classifier
        self.trainingNetwork.setLearningMode(learningSP=False,
                                             learningTM=False,
                                             learningTP=True,
                                             learningClassifier=True)
        self.trainingNetwork.resetIndex()
        threading.Thread(target=self.runNetworkBatch,
                         kwargs={"network": self.trainingNetwork,
                                 "queue": self.eventQueue,
                                 "process": "CLAS"}).start()
        self.root.after(100, self.processEventQueue)
      elif (msg["process"] == "TM" and
            msg["running"] == True):
        self.progressbarLearning.step()
        threading.Thread(target=self.runNetworkBatch,
                         kwargs={"network": self.trainingNetwork,
                                 "queue": self.eventQueue,
                                 "process": "TM"}).start()
        self.root.after(100, self.processEventQueue)

      # CLASSIFIER
      elif (msg["process"] == "CLAS" and
            msg["running"] == False):
        #self.trainingNetwork.saveNetwork()
        self.progressbarRunning.stop()
        self.progressbarLearning.stop()
        self.buttonLoadTesting.config(state=Tk.NORMAL)
      elif (msg["process"] == "CLAS" and
            msg["running"] == True):
        self.progressbarLearning.step()
        threading.Thread(target=self.runNetworkBatch,
                         kwargs={"network": self.trainingNetwork,
                                 "queue": self.eventQueue,
                                 "process": "CLAS"}).start()
        self.root.after(100, self.processEventQueue)

      # TESTING
      elif (msg["process"] == "TEST" and
            msg["running"] == False):
        print "Done testing!"
        tkMessageBox.showinfo("INFO",
                              "Classifier num correct = {num}".format(
                                  num=self.testingNetwork.numCorrect))
        self.progressbarRunning.stop()
        self.progressbarLearning.stop()
      elif (msg["process"] == "TEST" and
            msg["running"] == True):
        self.progressbarLearning.step()
        print ("Testing iteration: {iter}, % correct: {acc}"
               .format(iter=self.testingNetwork.testingImageIndex,
                       acc=(self.testingNetwork.numCorrect*100.0/
                            self.testingNetwork.testingImageIndex)))
        threading.Thread(target=self.testNetworkBatch,
                         kwargs={"network": self.testingNetwork,
                                 "queue": self.eventQueue,
                                 "process": "TEST"}).start()
        self.root.after(100, self.processEventQueue)

    except Queue.Empty:
      self.progressbarRunning.step()
      self.root.after(100, self.processEventQueue)



if __name__ == "__main__":
  datetimestr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  if not os.path.exists(_NETWORK_DIR_NAME):
    os.makedirs(_NETWORK_DIR_NAME)
  netName = "%s/%s_mnist_net.nta" % (_NETWORK_DIR_NAME, datetimestr)

  root = Tk.Tk()
  root.title("Saccades Experiment")
  root.geometry("{width}x{height}".format(width=_APP_WIDTH, height=_APP_HEIGHT))
  root.resizable(width=Tk.FALSE, height=Tk.FALSE)
  MainGUI(root,
          loggingDir=None,
          networkName=netName,
          trainingSet="data/supersmall_training",
          testingSet="data/testing")
  root.mainloop()

  exit()
