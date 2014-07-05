#! /usr/bin/env python
'''
This script trains the spatial pooler (SP) on a set of images that are 
listed in the XML file specified by trainingDataset.  The SP is trained 
for a maximum number of training cycles given by maxTrainingCycles and then its
classification abilities are tested on the images listed in the XML file
specified by testingDataset.
'''

trainingDataset = 'DataSets/OCR/characters/hex.xml'
maxTrainingCycles = 10
testingDataset = 'DataSets/OCR/characters/hex.xml'
print "Training data set: ",trainingDataset
print "Testing data set: ",testingDataset

import dataset_readers as data
import image_encoders as encoder
from nupic.research.spatial_pooler import SpatialPooler
from vision_testbench import VisionTestBench


# Get training images and convert them to vectors.
trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
trainingVectors = encoder.imagesToVectors(trainingImages)


# Instantiate the python spatial pooler
sp = SpatialPooler(
  inputDimensions = 32**2, # Size of image patch
  columnDimensions = 16, # Number of potential features
  potentialRadius = 10000, # Ensures 100% potential pool
  potentialPct = 1, # Neurons can connect to 100% of input
  globalInhibition = True,
  localAreaDensity = -1, # Using numActiveColumnsPerInhArea 
  #localAreaDensity = 0.02, # one percent of columns active at a time
  #numActiveColumnsPerInhArea = -1, # Using percentage instead
  numActiveColumnsPerInhArea = 1, # Only one feature active at a time
  # All input activity can contribute to feature output
  stimulusThreshold = 0,
  synPermInactiveDec = 0.1,
  synPermActiveInc = 0.1,
  synPermConnected = 0.1, # Connected threshold
  maxBoost = 3,
  seed = 1956, # The seed that Grok uses
  spVerbosity = 1)


# Instantiate the spatial pooler test bench.
tb = VisionTestBench(sp)

# Train the spatial pooler on trainingVectors.
SDRs, numCycles = tb.train(trainingVectors, trainingTags, maxTrainingCycles)

# View the permanences and connections after training.
tb.showPermsAndConns()
#tb.savePermsAndConns('perms_and_conns.jpg')

# Get testing images and convert them to vectors.
testingImages, testingTags = data.getImagesAndTags(testingDataset)
testingVectors = encoder.imagesToVectors(testingImages)

# Test the spatial pooler on testingVectors.
testSDRs = tb.test(testingVectors, testingTags)
if testSDRs != SDRs:
  print "Yo! SDRs don't match!"
  #for i in range(len(testSDRs)):
    #print "%5s %5s" % (SDRs[i], testSDRs[i])

# Classifier Hack, uses the testing image tags along with the SDRs from the 
# last training cycle to interpret the SDRs from testing.
testResults = []
[testResults.append('') for i in range(len(testSDRs))]
for i,testSDR in enumerate(testSDRs):
  for j,SDR in enumerate(SDRs):
    if testSDR == SDR:
      if len(testResults[i]) == 0:
        testResults[i] += trainingTags[j]
      elif trainingTags[j] not in testResults[i]:
        testResults[i] += "," + trainingTags[j]


# Show the test results
print "\nTest Results:\n"
print "%5s" % "Input", "Output"
for i in range(len(testingTags)):
  print "%-5s" % testingTags[i], testResults[i]
print 

accuracy = 0.0
for i in range(len(testResults)):
  if testingTags[i] == testResults[i]:
    accuracy += 100.0/len(testResults)

print "Accuracy: %.1f" % accuracy,"%"
print "Number of training cycles: ", numCycles
print 


