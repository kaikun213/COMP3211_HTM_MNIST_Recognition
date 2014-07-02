#! /usr/bin/env python
'''
This script trains the spatial pooler (SP) on a set of images that are 
listed in the XML file specified by trainingDataset.  The SP is trained 
for the number of training cycles given by trainingCycles and then its
classification abilities are tested on the images listed in the XML file
specified by testingDataset.
'''

trainingDataset = 'Datasets/OCR/characters/hex.xml'
maxTrainingCycles = 10
testingDataset = 'Datasets/OCR/characters/hex.xml'
print "Training data set: ",trainingDataset
print "Testing data set: ",testingDataset

import dataset_readers as data
import image_encoders as encoder
from parameters import Parameters
from nupic.research.spatial_pooler import SpatialPooler
#from nupic.encoders import ScalarEncoder
from vision_testbench import VisionTestBench


# Get training images and convert them to vectors.
trainingImages, trainingTags = data.getImagesAndTags(trainingDataset)
trainingVectors = encoder.imagesToVectors(trainingImages)


# Specify parameter values to search 
parameters = Parameters()
#parameters.define("numCols",[256])
parameters.define("numCols",[256,512,1024,2048])
parameters.define("synPermConn",[0.5])
#parameters.define("synPermConn",[0.1,0.3,0.5,0.7,0.9])
parameters.define("synPermIncFrac",[1.0])
#parameters.define("synPermIncFrac",[1.0,0.5,0.1])
parameters.define("synPermDecFrac",[1.0])
#parameters.define("synPermDecFrac",[1.0,0.5,0.1])


# Run the model until all combinations have been tried
combinations = []  # list for storing parameter combinations
results = []  # list for storing image recognition accuracy results 
while len(results) < parameters.combinations:
  
  # Pick a random combination of parameter values
  #parameters.generateRandomCombination()
  numCols = parameters.getValue("numCols")
  synPermConn = parameters.getValue("synPermConn")
  synPermDec = synPermConn*parameters.getValue("synPermDecFrac")
  synPermInc = synPermConn*parameters.getValue("synPermIncFrac")

  # Run it if it hasn't been tried yet
  if parameters.getAllValues() not in combinations:
    print "Parameter Combination: ", parameters.getAllValues()
    # Instantiate our spatial pooler
    sp = SpatialPooler(
      inputDimensions = 32**2, # Size of image patch
      columnDimensions = numCols, # Number of potential features
      potentialRadius = 10000, # Ensures 100% potential pool
      potentialPct = 1, # Neurons can connect to 100% of input
      globalInhibition = True,
      #localAreaDensity = -1, # Using numActiveColumnsPerInhArea 
      localAreaDensity = 0.02, # one percent of columns active at a time
      numActiveColumnsPerInhArea = -1, # Using percentage instead
      #numActiveColumnsPerInhArea = 1, # Only one feature active at a time
      # All input activity can contribute to feature output
      stimulusThreshold = 0,
      synPermInactiveDec = synPermDec,
      synPermActiveInc = synPermInc,
      synPermConnected = synPermConn, # Connected threshold
      maxBoost = 3,
      seed = 1956, # The seed that Grok uses
      spVerbosity = 1)
    
    
    # Instantiate the spatial pooler test bench.
    tb = VisionTestBench(sp)
    
    # Train the spatial pooler on trainingVectors.
    SDRs, numCycles = tb.train(trainingVectors, trainingTags, maxTrainingCycles)

    # Save the permanences and connections after training.
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
    #print "\nTest Results:\n"
    #print "%5s" % "Input", "Output"
    #for i in range(len(testingTags)):
      #print "%-5s" % testingTags[i], testResults[i]
    #print 
    
    accuracy = 0.0
    for i in range(len(testResults)):
      if testingTags[i] == testResults[i]:
        accuracy += 100.0/len(testResults)
    
    combinations.append(parameters.getAllValues()[:])  # pass list by value
    results.append([accuracy, numCycles])
    
    print 
    print "Parameter Combination: ", parameters.getAllValues()
    print "Accuracy: %.1f" % accuracy,"%"
    print "Number of training cycles: ", numCycles
    print 
    print "Combinations completed: ", len(combinations), "/", parameters.combinations
    print 

  # Try next parameter combination
  parameters.generateNextCombination()
    
    
for i in range(len(combinations)):
  print combinations[i], "%.1f" % results[i][0],"% ", results[i][1],"cycles"
    
    
