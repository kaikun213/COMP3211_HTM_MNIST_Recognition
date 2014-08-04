'''
This script trains the spatial pooler (SP) on a set of images until it achieves
either a minimum specified image recognition accuracy on the training data set
or until a maximum number of training cycles is reached.  
Then its image recognition abilities are tested on another set of images. 

trainingDataset - name of XML file that lists the training images

testingDataset - name of XML file that lists the testing images

minAccuracy - minimum accuracy requred to stop training before 
              maxTrainingCycles is reached

maxTrainingCycles - maximum number of training cycles to perform


'''

trainingDataset = 'DataSets/OCR/characters/all.xml'
testingDataset = 'DataSets/OCR/characters/all.xml'
minAccuracy = 200.0
maxTrainingCycles = 2

import dataset_readers as data
import image_encoders as encoder
from parameters import Parameters
from nupic.research.spatial_pooler import SpatialPooler
#from nupic.bindings.algorithms import SpatialPooler
from vision_testbench import VisionTestBench
from classifiers import exactMatch
from classifiers import KNNClassifier


# Get training images and convert them to vectors.
trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
trainingVectors = encoder.imagesToVectors(trainingImages)

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
  maxBoost = 1.0,
  seed = 1956, # The seed that Grok uses
  spVerbosity = 1)

# Instantiate the spatial pooler test bench.
tb = VisionTestBench(sp)

# Instantiate the classifier
#clf = exactMatch()
clf = KNNClassifier()

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
#print testingTags
#testingTags = [testingTag for testingTag in reversed(testingTags)]
#print testingTags
#print testingVectors[0].sum()
#testingVectors = [testingVector for testingVector in reversed(testingVectors)]
#print testingVectors[-1].sum()

# Test the spatial pooler on testingVectors.
accuracy = tb.test(testingVectors, testingTags, clf)

print "Number of training cycles:", numCycles
print "Accuracy:", accuracy


