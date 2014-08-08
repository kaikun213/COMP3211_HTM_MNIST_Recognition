'''
This script trains and tests the spatial pooler (SP) over and over while varying
some SP parameters.  For each set of SP parameter values it trains the spatial
pooler (SP) on a set of training images and tests its image recognition
abilities on a set of test images. 

During training the SP is trained until it achieves either a minimum specified
image recognition accuracy on the training data set or until a maximum number of
training cycles is reached.  

After training and testing is completed for all combinations of the parameter
values, a summary of the results is displayed.


trainingDataset - name of XML file that lists the training images

testingDataset - name of XML file that lists the testing images

minAccuracy - minimum accuracy requred to stop training before 
              maxTrainingCycles is reached

maxTrainingCycles - maximum number of training cycles to perform


'''

trainingDataset = 'DataSets/OCR/characters/hex.xml'
minAccuracy = 100.0
maxTrainingCycles = 5
testingDataset = 'DataSets/OCR/characters/hex.xml'

import dataset_readers as data
import image_encoders as encoder
from parameters import Parameters
from nupic.research.spatial_pooler import SpatialPooler
from vision_testbench import VisionTestBench
from classifiers import KNNClassifier


# Get training images and convert them to vectors.
trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
trainingVectors = encoder.imagesToVectors(trainingImages)


# Specify parameter values to search
parameters = Parameters()
parameters.define("synPermConn", [0.5])
parameters.define("synPermDecFrac", [1.0, 0.5, 0.1])
parameters.define("synPermIncFrac", [1.0, 0.5, 0.1])


# Run the model until all combinations have been tried
while parameters.getNumResults() < parameters.numCombinations:

  # Pick a combination of parameter values
  parameters.nextCombination()
  #parameters.nextRandomCombination()
  synPermConn = parameters.getValue("synPermConn")
  synPermDec = synPermConn*parameters.getValue("synPermDecFrac")
  synPermInc = synPermConn*parameters.getValue("synPermIncFrac")

  # Instantiate our spatial pooler
  sp = SpatialPooler(
    inputDimensions= (32, 32), # Size of image patch
    columnDimensions = (32, 32),
    potentialRadius = 10000, # Ensures 100% potential pool
    potentialPct = 0.8,
    globalInhibition = True,
    localAreaDensity = -1, # Using numActiveColumnsPerInhArea
    numActiveColumnsPerInhArea = 64,
    # All input activity can contribute to feature output
    stimulusThreshold = 0,
    synPermInactiveDec = synPermDec,
    synPermActiveInc = synPermInc,
    synPermConnected = synPermConn,
    maxBoost = 1.0,
    seed = 1956, # The seed that Grok uses
    spVerbosity = 1)

  # Instantiate the spatial pooler test bench.
  tb = VisionTestBench(sp)

  # Instantiate the classifier
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
  testingTags = [testingTag for testingTag in reversed(testingTags)]
  testingVectors = [testingVector for testingVector in reversed(testingVectors)]

  # Test the spatial pooler on testingVectors.
  accuracy = tb.test(testingVectors, testingTags, clf)

  # Add results to the list
  parameters.appendResults([accuracy, numCycles])


parameters.printResults(["Percent Accuracy", "Training Cycles"], [", %.2f", ", %d"])
print "The maximum number of training cycles is set to:", maxTrainingCycles


