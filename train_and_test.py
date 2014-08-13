'''
This script trains the spatial pooler (SP) on a set of images until it achieves
either a minimum specified image recognition accuracy on the training data set
or until a maximum number of training cycles is reached.  Then its image 
recognition abilities are tested on another set of images. 

trainingDataset - name of XML file that lists the training images

testingDataset - name of XML file that lists the testing images

minAccuracy - minimum accuracy requred to stop training before 
              maxTrainingCycles is reached

maxTrainingCycles - maximum number of training cycles to perform

'''

trainingDataset = 'DataSets/OCR/characters/cmr_all.xml'
testingDataset = 'DataSets/OCR/characters/cmr_all.xml'
minAccuracy = 100.0  # force max training cycles
maxTrainingCycles = 5

import dataset_readers as data
import image_encoders as encoder
from nupic.research.spatial_pooler import SpatialPooler as SpatialPooler
#from nupic.bindings.algorithms import SpatialPooler as SpatialPooler
from vision_testbench import VisionTestBench
from classifiers import KNNClassifier

from nupic.support.unittesthelpers.algorithm_test_helpers import convertSP

import numpy # just for debugging

# Instantiate our spatial pooler
sp = SpatialPooler(
  inputDimensions= (32, 32), # Size of image patch
  columnDimensions = (32, 32),
  potentialRadius = 10000, # Ensures 100% potential pool
  potentialPct = 0.8,
  globalInhibition = True,
  localAreaDensity = -1, # Using numActiveColumnsPerInhArea
  #localAreaDensity = 0.02, # one percent of columns active at a time
  #numActiveColumnsPerInhArea = -1, # Using percentage instead
  numActiveColumnsPerInhArea = 64,
  # All input activity can contribute to feature output
  stimulusThreshold = 0,
  synPermInactiveDec = 0.001,
  synPermActiveInc = 0.001,
  synPermConnected = 0.3,
  minPctOverlapDutyCycle=0.001,
  minPctActiveDutyCycle=0.001,
  dutyCyclePeriod=1000,
  maxBoost = 1.0,
  seed = 1956, # The seed that Grok uses
  spVerbosity = 1)

# Instantiate the spatial pooler test bench.
tb = VisionTestBench(sp)

# Instantiate the classifier
clf = KNNClassifier()

# Get training images and convert them to vectors.
trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
trainingVectors = encoder.imagesToVectors(trainingImages)

# Train the spatial pooler on trainingVectors.
numCycles = tb.train(trainingVectors, trainingTags, clf, maxTrainingCycles,
  minAccuracy)

# Save the permanences and connections after training.
#tb.savePermanences('perms.jpg')
#tb.showPermanences()
#tb.showConnections()

# Get testing images and convert them to vectors.
testingImages, testingTags = data.getImagesAndTags(testingDataset)
testingVectors = encoder.imagesToVectors(testingImages)

# Reverse the order of the vectors and tags for testing
testingTags = [testingTag for testingTag in reversed(testingTags)]
testingVectors = [testingVector for testingVector in reversed(testingVectors)]

# Test the spatial pooler on testingVectors.
accuracy = tb.test(testingVectors, testingTags, clf, learn=True)
print "Number of training cycles:", numCycles


