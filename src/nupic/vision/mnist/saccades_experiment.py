# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

"""Saccading vision demo with MNIST."""

import pkg_resources

import capnp
from nupic.proto import ExtendedTemporalMemoryProto_capnp
from nupic.vision.mnist.saccade_network import SaccadeNetwork

TRAIN_IMAGES = pkg_resources.resource_filename("nupic.vision.mnist", "mnist/small_training")
TEST_IMAGES = pkg_resources.resource_filename("nupic.vision.mnist", "mnist/small_training")
#LOG_DIR = pkg_resources.resource_filename("nupic.vision.mnist", "logs")
LOG_DIR = None



def createNetwork():
  trainingNetwork = SaccadeNetwork(
      networkName="SaccadeNetwork",
      trainingSet=TRAIN_IMAGES, validationSet=None,
      testingSet=TEST_IMAGES, loggingDir=LOG_DIR,
      createNetwork=True)
  return trainingNetwork



if __name__ == "__main__":
  net = createNetwork()

  # Training
  numTrain = 1

  # train SP
  net.loadExperiment()
  net.setLearningMode(learningSP=True,
                      learningTM=False,
                      learningTP=False,
                      learningClassifier=False)
  for i in xrange(numTrain):
    print "Running SP train batch #{}".format(i)
    net.runNetworkBatch(10)
  print "Train index: {}".format(net.trainingImageIndex)

  # train TM
  net.loadExperiment()
  net.setLearningMode(learningSP=False,
                      learningTM=True,
                      learningTP=False,
                      learningClassifier=False)
  for i in xrange(numTrain):
    print "Running TM train batch #{}".format(i)
    net.runNetworkBatch(10)
  print "Train index: {}".format(net.trainingImageIndex)

  # train classifier
  net.loadExperiment()
  net.setLearningMode(learningSP=False,
                      learningTM=False,
                      learningTP=True,
                      learningClassifier=True)
  for i in xrange(numTrain):
    print "Running classifier train batch #{}".format(i)
    net.runNetworkBatch(10)
  print "Train index: {}".format(net.trainingImageIndex)

  # Testing
  numTest = 10
  net.loadExperiment()
  net.setLearningMode(learningSP=False,
                      learningTM=False,
                      learningTP=False,
                      learningClassifier=False)
  net.setupNetworkTest()
  numCorrect = net.testNetworkBatch(numTest)
  print "Got {} correct out of {}".format(numCorrect, numTest)
  print "Train index: {}".format(net.trainingImageIndex)
