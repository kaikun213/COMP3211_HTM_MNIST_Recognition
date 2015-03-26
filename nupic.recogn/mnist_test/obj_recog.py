#/usr/bin/env python
# It's a test program for image read in python

import os,sys
import Image
import numpy as np
from nupic.research.TP10X2 import TP10X2 as TP
from nupic.research.spatial_pooler import SpatialPooler as SP
class Object_Recognition():
   """A test project for object recognition implemented by nupic sp & tp."""
   
   def __init__(self,
                inputShape = (28, 28),
                columnDimensions = (16, 16),
                activeRate = 0.1, precision = 0.8):
      
      self.activeRate = activeRate
      self.inputShape = inputShape
      self.columnDimensions = columnDimensions
      self.inputSize = np.array(inputShape).prod()
      self.columnNumber = np.array(columnDimensions).prod()
      self.inputArray = np.zeros(self.inputSize)
      self.precision = precision

      self.sp = SP(self.inputShape,
                   self.columnDimensions,
                   potentialRadius = self.inputSize,
                   potentialPct = 1.0,
                   numActiveColumnsPerInhArea = int(activeRate*self.columnNumber),
                   globalInhibition = True,
                   synPermActiveInc = 0, #1.0/(2**precision),
                   stimulusThreshold = 0,
                   synPermInactiveDec = 0, #1.0/(2**precision),
                   synPermConnected = 0.7,
                   maxBoost = 1.0,
                   spVerbosity = 0
                  )
      self.activeArray = np.zeros(self.sp.getNumColumns())
      self.tp = TP(numberOfCols=self.inputSize, cellsPerColumn=4,
                   initialPerm=0.5, connectedPerm=0.5,
                   minThreshold=10, newSynapseCount=10,
                   permanenceInc=0.1, permanenceDec=0.0,
                   activationThreshold=8,
                   globalDecay=0, burnIn=1,
                   checkSynapseConsistency=False,
                   pamLength=10)
      self.input_bin = np.zeros(self.inputSize)
      self.sp_out    = np.zeros(self.columnNumber)

   def spatialPooling(self, image, Learn = False,\
                      verbosity=0):
      self.sp.compute(image, Learn, self.activeArray)
      if verbosity:
         print self.activeArray
      #if activeList not in self.SDRs:
      #   self.SDRs.append(activeList)
      #   SDRI = self.SDRs.index(activeList)
      #   self.SDRIs.append(SDRI)
      #category = trainingTags.index(trainingTags[j])
      #classifier.learn(self.activeArray, category)
      return self.activeArray

   def temporalPooling(self, Learn=False):
      self.tp.compute(self.sp_out, enableLearn = Learn,  computeInfOutput = False)

#   def spTpReset(self):
#      self.sp.reset()
#      self.tp.reset()

   def spTest(self, testImage, classifier, groundTruth):
      self.sp.compute(testImage, False, self.activeArray)
      inferred = classifier.infer(self.activeArray)[0]
      if inferred == groundTruth:
         return True, inferred
      else:
         return False, inferred
   def printSPStatus(self):
      self.sp.printParameters()
