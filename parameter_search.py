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

trainingDataset = 'DataSets/OCR/characters/hex.xml'
minAccuracy = 200.0
maxTrainingCycles = 5
testingDataset = 'DataSets/OCR/characters/hex.xml'

import dataset_readers as data
import image_encoders as encoder
from parameters import Parameters
from nupic.research.spatial_pooler import SpatialPooler
from vision_testbench import VisionTestBench
from classifiers import exactMatch
from classifiers import KNNClassifier


# Get training images and convert them to vectors.
trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
trainingVectors = encoder.imagesToVectors(trainingImages)


# Specify parameter values to search
parameters = Parameters()
#parameters.define("dataSet",['32.xml'])
parameters.define("dataSet",[
  '1.xml','2.xml', '3.xml', '4.xml', '5.xml', '6.xml', '7.xml', '8.xml',
  '9.xml', '10.xml', '11.xml', '12.xml', '13.xml', '14.xml', '15.xml',
  '16.xml', '17.xml', '18.xml', '19.xml', '20.xml', '21.xml', '22.xml',
  '23.xml', '24.xml', '25.xml', '26.xml', '27.xml', '28.xml', '29.xml',
  '30.xml', '31.xml', '32.xml', '33.xml', '34.xml', '35.xml', '36.xml',
  '37.xml', '38.xml', '39.xml', '40.xml', '41.xml', '42.xml', '43.xml',
  '44.xml', '45.xml', '46.xml', '47.xml', '48.xml', '49.xml', '50.xml',
  '51.xml', '52.xml', '53.xml', '54.xml', '55.xml', '56.xml', '57.xml',
  '58.xml', '59.xml', '60.xml', '61.xml', '62.xml'])
#parameters.define("numCols", [(32, 32)])
#parameters.define("numCols", [256,512,1024,2048])
#parameters.define("synPermConn", [0.3])
#parameters.define("synPermConn", [0.9, 0.7, 0.5, 0.3, 0.1])
#parameters.define("synPermDecFrac", [1.0])
#parameters.define("synPermDecFrac", [1.0, 0.5, 0.1])
#parameters.define("synPermIncFrac", [1.0])
#parameters.define("synPermIncFrac", [1.0, 0.5, 0.1])


# Run the model until all combinations have been tried
while parameters.getNumResults() < parameters.numCombinations:

  # Pick a combination of parameter values
  parameters.nextCombination()
  dataSet = parameters.getValue("dataSet")
  trainingDataset = 'DataSets/OCR/characters/' + dataSet
  trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
  trainingVectors = encoder.imagesToVectors(trainingImages)
  testingDataset = 'DataSets/OCR/characters/' + dataSet
  #parameters.nextRandomCombination()
  #numCols = parameters.getValue("numCols")
  #synPermConn = parameters.getValue("synPermConn")
  #synPermDec = synPermConn*parameters.getValue("synPermDecFrac")
  #synPermInc = synPermConn*parameters.getValue("synPermIncFrac")

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

  # Add results to the list
  parameters.appendResults([accuracy, numCycles])


parameters.printResults(["Percent Accuracy", "Training Cycles"], [", %.2f", ", %d"])
print "The maximum number of training cycles is set to:", maxTrainingCycles


