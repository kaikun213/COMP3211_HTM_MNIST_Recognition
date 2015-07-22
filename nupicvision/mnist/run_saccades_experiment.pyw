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

import datetime
import os
import Queue
import threading
import Tkinter as Tk
import tkMessageBox
import ttk as Ttk

from saccade_network import (
    SaccadeNetwork,
    SACCADES_PER_IMAGE,
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



class MainGUI(object):
  """ A GUI for running the saccade experiment. Allows the user to run the
  experiment with visualizations of individual saccades and use this to tune
  the parameters for making saccades.
  """


  def __init__(self, root,
               loggingDir,
               networkName,
               trainingSet="mnist/small_training",
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
    for i in range(SACCADES_PER_IMAGE):
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


    # Add images (labels) to saccade detail frame
    self.saccadeHistImgList = [None for i in range(SACCADES_PER_IMAGE)]
    self.saccadeHistImg = Tk.Label(self.saccadeDetailImagesFrame,
                                   height=_APP_DETAILED_SACCADE_HEIGHT,
                                   width=_APP_DETAILED_SACCADE_WIDTH)
    self.saccadeHistImg.image = None #self.saccadeHistImg
    self.saccadeHistLabel = Tk.Label(self.saccadeDetailImagesFrame,
                                     text="History")
    self.saccadeDetailImgList = [None for i in range(SACCADES_PER_IMAGE)]
    self.saccadeCurImage = Tk.Label(self.saccadeDetailImagesFrame,
                                    height=_APP_DETAILED_SACCADE_HEIGHT,
                                    width=_APP_DETAILED_SACCADE_WIDTH)
    self.saccadeCurImage.image = None
    self.saccadeCurLabel = Tk.Label(self.saccadeDetailImagesFrame,
                                    text="Current saccade")

    self.saccadeHistImg.pack(side=Tk.TOP, expand=Tk.YES, fill=Tk.X)
    self.saccadeHistLabel.pack(side=Tk.TOP, fill=Tk.X)
    self.saccadeCurImage.pack(side=Tk.TOP, expand=Tk.YES, fill=Tk.X)
    self.saccadeCurLabel.pack(side=Tk.TOP, fill=Tk.X)


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
    self.buttonLoadExperiment = Tk.Button(self.controlButtonsFrame,
                                          text="Load...",
                                          command=self.buttonLoadExperimentCb)
    self.buttonRun = Tk.Button(self.controlButtonsFrame,
                               text="Run",
                               command=self.buttonRunCb,
                               state=Tk.DISABLED)
    self.enableVisualizations = Tk.IntVar()
    self.enableVisualizationsCheckbox = Tk.Checkbutton(
        self.controlButtonsFrame, text="Show visualizations",
        variable=self.enableVisualizations)
    self.enableVisualizationsCheckbox.select()
    self.buttonLoadExperiment.pack(side=Tk.LEFT)
    self.buttonRun.pack(side=Tk.LEFT)
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

    # Set up the network & queue
    self.eventQueue = Queue.Queue()
    self.network = SaccadeNetwork(
        loggingDir=loggingDir, networkName=networkName,
        trainingSet=trainingSet, validationSet=validationSet,
        testingSet=testingSet,
        detailedSaccadeWidth=_APP_DETAILED_SACCADE_WIDTH,
        detailedSaccadeHeight=_APP_DETAILED_SACCADE_HEIGHT,
        createNetwork=False)
    print "Ready..."


  def buttonLoadExperimentCb(self):
    """ Callback for the "Load" button. Creates a network and loads
    the training images for the experiment.
    """
    self.network.createNet()
    self.network.loadExperiment()

    # Update GUI
    self.buttonLoadExperiment.config(state=Tk.DISABLED)
    self.buttonRun.config(state=Tk.NORMAL)
    self.buttonNextImg.config(state=Tk.NORMAL)
    self.buttonNextImgCb()


  def buttonNextImgCb(self):
    """ Callback for the "Next" button. Advances the network one training
    image.
    """
    if self.enableVisualizations.get():
      result = self.network.runNetworkOneImage(enableViz=True)

      if result == False:
        # The sp has no more images to learn
        self.buttonNextImg.config(state=Tk.DISABLED)
        self.buttonRun.config(state=Tk.DISABLED)
        tkMessageBox.showinfo("INFO", "SP learning is done!")
        return

      (saccadeImgs, saccadeDetailImgs, saccadeHistImgs) = result
      for i in range(SACCADES_PER_IMAGE):
        self.saccadeImgs[i].configure(image=saccadeImgs[i])
        self.saccadeImgs[i].image = saccadeImgs[i]
        self.saccadeImgs[i].configure(bg="blue")

        self.saccadeHistImgList[i] = saccadeHistImgs[i]
        self.saccadeDetailImgList[i] = saccadeDetailImgs[i]

      self.currentDetailSaccadeIndex = SACCADES_PER_IMAGE-1 # Show last saccade
      self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="red")
      self.buttonScrubFirst.config(state=Tk.NORMAL)
      self.buttonScrubRev.config(state=Tk.NORMAL)
      self.buttonScrubLast.config(state=Tk.DISABLED)
      self.buttonScrubFwd.config(state=Tk.DISABLED)
      self.updateDetailImages()

    else:
      result = self.network.runNetworkOneImage(enableViz=False)

      if result == False:
        self.buttonNextImg.config(state=Tk.DISABLED)
        self.buttonRun.config(state=Tk.DISABLED)
        tkMessageBox.showinfo("INFO", "SP learning is done!")
        return


  def buttonScrubFirstCb(self):
    """ Callback for the "First" transport button. Go to the first saccade
    into the detailed saccade section """
    self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="blue")
    self.currentDetailSaccadeIndex = 0
    self.buttonScrubFirst.config(state=Tk.DISABLED)
    self.buttonScrubRev.config(state=Tk.DISABLED)
    self.buttonScrubLast.config(state=Tk.NORMAL)
    self.buttonScrubFwd.config(state=Tk.NORMAL)
    self.updateDetailImages()


  def buttonScrubRevCb(self):
    """ Callback for the "Reverse" transport button. Go back by one saccade
    in the detailed saccade section """
    self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="blue")
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
    self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="blue")
    self.currentDetailSaccadeIndex +=1
    if self.currentDetailSaccadeIndex == SACCADES_PER_IMAGE-1:
      self.buttonScrubLast.config(state=Tk.DISABLED)
      self.buttonScrubFwd.config(state=Tk.DISABLED)
    self.buttonScrubFirst.config(state=Tk.NORMAL)
    self.buttonScrubRev.config(state=Tk.NORMAL)
    self.updateDetailImages()


  def buttonScrubLastCb(self):
    """ Callback for the "Last" transport button. Go to the last saccade
    in the detailed saccade section """
    self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="blue")
    self.currentDetailSaccadeIndex = SACCADES_PER_IMAGE-1
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
    self.saccadeImgs[self.currentDetailSaccadeIndex].configure(bg="red")


  def buttonRunCb(self):
    """ Runs the training on the network.
    NOTE: Automatically disables visualizations for speed
    """
    self.enableVisualizationsCheckbox.deselect()
    self.buttonNextImg.config(state=Tk.DISABLED)
    self.buttonRun.config(state=Tk.DISABLED)
    self.buttonScrubLast.config(state=Tk.DISABLED)
    self.buttonScrubFwd.config(state=Tk.DISABLED)
    self.buttonScrubFirst.config(state=Tk.DISABLED)
    self.buttonScrubRev.config(state=Tk.DISABLED)

    self.progressbarRunning.step()

    threading.Thread(target=self.runNetworkBatch,
                     kwargs={"network": self.network,
                             "queue": self.eventQueue}).start()

    self.root.after(100, self.processEventQueue)


  @staticmethod
  def runNetworkBatch(network, queue):
    """ A static function that allows the network to be run in batches from
    threads that won't block the GUI from updating.

    :param network: An initialized network
    :param queue: The queue to send results to
    """
    result = network.runNetworkBatch(batchSize=10)
    queue.put({"process": "runNetworkBatch",
               "running": result})


  def processEventQueue(self):
    """ GUI method periodically called when running large jobs on the network.
    Updates progress bars with network training/testing progress.
    """
    try:
      msg = self.eventQueue.get(0)
      if (msg["process"] == "runNetworkBatch" and
          msg["running"] == False):
        tkMessageBox.showinfo("INFO", "SP learning is done!")
        self.progressbarRunning.stop()
        self.progressbarLearning.stop()
      elif (msg["process"] == "runNetworkBatch" and
            msg["running"] == True):
        self.progressbarLearning.step()
        threading.Thread(target=self.runNetworkBatch,
                         kwargs={"network": self.network,
                                 "queue": self.eventQueue}).start()
        self.root.after(100, self.processEventQueue)
    except Queue.Empty:
      self.progressbarRunning.step()
      self.root.after(100, self.processEventQueue)



if __name__ == "__main__":
  datetimestr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  networkDirName = "networks"
  if not os.path.exists(networkDirName):
    os.makedirs(networkDirName)
  netName = "%s/%s_mnist_net.nta" % (networkDirName, datetimestr)

  root = Tk.Tk()
  root.title("Saccades Experiment")
  root.geometry("{width}x{height}".format(
      width=_APP_WIDTH, height=_APP_HEIGHT))
  root.resizable(width=Tk.FALSE, height=Tk.FALSE)
  MainGUI(root,
          loggingDir=None, #datetimestr,
          networkName=netName,
          trainingSet="mnist/small_training",
          testingSet="mnist/testing")
  root.mainloop()

  exit()
