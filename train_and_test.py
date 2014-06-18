#! /usr/bin/env python
'''
This script trains the spatial pooler (SP) on a set of images that are 
listed in the XML file specified by trainingDataset.  The SP is trained 
for the number of training cycles given by trainingCycles and then its
classification abilities are tested on the images listed in the XML file
specified by testingDataset.
'''

trainingDataset = 'Datasets/OCR/small_test.xml'
trainingCycles = 20
testingDataset = 'Datasets/OCR/small_test.xml'

import dataset_readers as data
import image_encoders as encoder
from nupic.research.spatial_pooler import SpatialPooler
#from nupic.encoders import ScalarEncoder
from vision_testbench import VisionTestBench


# Get training images and convert them to vectors.
trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
trainingVectors = encoder.imagesToVectors(trainingImages)


# Specify parameter ranges and resolutions to search
  
# Setup a table for all possible parameter combinations

# Pick a random combination of parameters

# If the table entry for this combination is empty then run the model
  
# Instantiate our spatial pooler
sp = SpatialPooler(
  inputDimensions = 32**2, # Size of image patch
  columnDimensions = 16, # Number of potential features
  potentialRadius = 10000, # Ensures 100% potential pool
  potentialPct = 1, # Neurons can connect to 100% of input
  globalInhibition = True,
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
for cycle in range(trainingCycles):
  SDRs = tb.train(trainingVectors, trainingTags)

# Save the permanences and connections after training.
tb.savePermsAndConns('perms_and_conns.jpg')

# Get testing images and convert them to vectors.
testingImages, testingTags = data.getImagesAndTags(testingDataset)
testingVectors = encoder.imagesToVectors(testingImages)

# Test the spatial pooler on testingVectors.
testSDRs = tb.test(testingVectors, testingTags)

# Classifier Hack, uses the testing image tags along with the SDRs from the 
# last training cycle to interpret the SDRs from testing.
testResults = []
[testResults.append('') for i in range(len(trainingTags))]
for i,testSDR in enumerate(testSDRs):
  for j,SDR in enumerate(SDRs):
    if testSDR == SDR:
      if len(testResults[i]) == 0:
        testResults[i] += trainingTags[j]
      else:
        testResults[i] += "," + trainingTags[j]

# Show the test results
print "\nTest Results:\n"
accuracy = 0.0
for i in range(len(SDRs)):
  if trainingTags[i] == testResults[i]:
    accuracy += 100.0/len(trainingTags)

print "%5s" % "Input", "Output"
for i in range(len(trainingTags)):
  print "%-5s" % trainingTags[i], testResults[i]
print 

print 
print "Accuracy:",accuracy,"%"
print 









