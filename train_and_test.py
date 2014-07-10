'''
This script trains the spatial pooler (SP) on a set of images that are
listed in the XML file specified by trainingDataset.  The SP is trained
for a maximum number of training cycles given by maxTrainingCycles and then its
classification abilities are tested on the images listed in the XML file
specified by testingDataset.
'''

trainingDataset = 'DataSets/OCR/characters/hex.xml'
minAccuracy = 200.0
maxTrainingCycles = 5
testingDataset = 'DataSets/OCR/characters/hex.xml'
print "Training data set: ", trainingDataset
print "Testing data set: ", testingDataset

#import numpy as np
import dataset_readers as data
import image_encoders as encoder
from parameters import Parameters
from nupic.research.spatial_pooler import SpatialPooler
#from nupic.encoders import ScalarEncoder
from vision_testbench import VisionTestBench
from classifiers import exactMatch
from classifiers import KNNClassifier


# Get training images and convert them to vectors.
trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
trainingVectors = encoder.imagesToVectors(trainingImages)


# Specify parameter values to search
parameters = Parameters()
parameters.define("dataSet",['62.xml'])
#parameters.define("dataSet",[
#  '1.xml','2.xml', '3.xml', '4.xml', '5.xml', '6.xml', '7.xml', '8.xml',
#  '9.xml', '10.xml', '11.xml', '12.xml', '13.xml', '14.xml', '15.xml',
#  '16.xml', '17.xml', '18.xml', '19.xml', '20.xml', '21.xml', '22.xml',
#  '23.xml', '24.xml', '25.xml', '26.xml', '27.xml', '28.xml', '29.xml',
#  '30.xml', '31.xml', '32.xml', '33.xml', '34.xml', '35.xml', '36.xml',
#  '37.xml', '38.xml', '39.xml', '40.xml', '41.xml', '42.xml', '43.xml',
#  '44.xml', '45.xml', '46.xml', '47.xml', '48.xml', '49.xml', '50.xml',
#  '51.xml', '52.xml', '53.xml', '54.xml', '55.xml', '56.xml', '57.xml',
#  '58.xml', '59.xml', '60.xml', '61.xml', '62.xml'])
#parameters.define("numCols", [(32, 32)])
#parameters.define("numCols", [256,512,1024,2048])
#parameters.define("synPermConn", [0.3])
#parameters.define("synPermConn", [0.9, 0.7, 0.5, 0.3, 0.1])
#parameters.define("synPermDecFrac", [1.0])
#parameters.define("synPermDecFrac", [1.0, 0.5, 0.1])
#parameters.define("synPermIncFrac", [1.0])
#parameters.define("synPermIncFrac", [1.0, 0.5, 0.1])


# Run the model until all combinations have been tried
combinations = []  # list for storing parameter combinations
results = []  # list for storing image recognition accuracy results
while len(results) < parameters.combinations:

  dataSet = parameters.getValue("dataSet")
  trainingDataset = 'DataSets/OCR/characters/' + dataSet
  trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
  trainingVectors = encoder.imagesToVectors(trainingImages)
  testingDataset = 'DataSets/OCR/characters/' + dataSet

  # Pick a random combination of parameter values
  #parameters.generateRandomCombination()
  #numCols = parameters.getValue("numCols")
  #synPermConn = parameters.getValue("synPermConn")
  #synPermDec = synPermConn*parameters.getValue("synPermDecFrac")
  #synPermInc = synPermConn*parameters.getValue("synPermIncFrac")

  # Run it if it hasn't been tried yet
  if parameters.getAllValues() not in combinations:
    print "\nParameter Combination: ", parameters.getAllValues()
    print
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
      synPermInactiveDec = 0.1,
      synPermActiveInc = 0.1,
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
    print "\nParameter Combination: ", parameters.getAllValues()
    numCycles = tb.train(trainingVectors, trainingTags, clf, maxTrainingCycles,
      minAccuracy)

    # Save the permanences and connections after training.
    #tb.savePermsAndConns('perms_and_conns.jpg')
    tb.showPermanences()

    # Get testing images and convert them to vectors.
    testingImages, testingTags = data.getImagesAndTags(testingDataset)
    testingVectors = encoder.imagesToVectors(testingImages)

    # Reverse the order of the vectors and tags for testing
    testingTags = [testingTag for testingTag in reversed(testingTags)]
    testingVectors = [testingVector for testingVector in reversed(testingVectors)]

    # Test the spatial pooler on testingVectors.
    print "\nParameter Combination: ", parameters.getAllValues()
    accuracy = tb.test(testingVectors, testingTags, clf)

    # Add results to the list
    combinations.append(parameters.getAllValues()[:])  # pass list by value
    results.append([accuracy, numCycles])

    print
    print "Parameter combinations completed: ",
    print len(combinations), "/", parameters.combinations
    print

  # Try next parameter combination
  parameters.generateNextCombination()


print "Summary of Results"
print
print "The maximum number of training cycles is set to:", maxTrainingCycles
print
headerList = parameters.getAllNames()
headerList.append("% Accuracy")
headerList.append("Training Cycles")
headerString = ", ".join(headerList)
print headerString
for i in range(len(combinations)):
  valueString = str(combinations[i])[1:-1]
  valueString += ", %.2f" % results[i][0]
  valueString += ", %d" % results[i][1]
  print valueString


