#! /usr/bin/env python
'''
This script trains the spatial pooler (SP) on a set of images that are 
listed in the XML file specified by trainingDataset.  The SP is trained 
for a maximum number of training cycles given by maxTrainingCycles and then its
classification abilities are tested on the images listed in the XML file
specified by testingDataset.
'''

trainingDataset = 'Datasets/OCR/characters/all.xml'
maxTrainingCycles = 30
testingDataset = 'Datasets/OCR/characters/all.xml'
print "Training data set: ",trainingDataset
print "Testing data set: ",testingDataset

import numpy as np
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
parameters.define("numCols",[256])
#parameters.define("numCols",[256,512,1024,2048])
#parameters.define("synPermConn",[0.5])
parameters.define("synPermConn",[0.9,0.7,0.5,0.3,0.1])
#parameters.define("synPermDecFrac",[1.0])
parameters.define("synPermDecFrac",[1.0,0.5,0.1])
#parameters.define("synPermIncFrac",[1.0])
parameters.define("synPermIncFrac",[1.0,0.5,0.1])


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
      localAreaDensity = -1, # Using numActiveColumnsPerInhArea 
      #localAreaDensity = 0.02, # one percent of columns active at a time
      #numActiveColumnsPerInhArea = -1, # Using percentage instead
      numActiveColumnsPerInhArea = 1,
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
    trainSDRIs, numCycles = tb.train(trainingVectors, trainingTags, maxTrainingCycles)

    # Save the permanences and connections after training.
    #tb.savePermsAndConns('perms_and_conns.jpg')
    #tb.showPermsAndConns()
    
    # Get testing images and convert them to vectors.
    testingImages, testingTags = data.getImagesAndTags(testingDataset)
    testingVectors = encoder.imagesToVectors(testingImages)
    
    # Test the spatial pooler on testingVectors.
    testSDRIs = tb.test(testingVectors, testingTags)

    if testSDRIs != trainSDRIs:
      print "Yo! SDRs don't match!"
      #for i in range(len(testSDRIs)):
        #if testSDRIs[i] != trainSDRIs[i]:
          #print "%6s %6s %6s" % (i, trainSDRIs[i], testSDRIs[i])
          #tb.printSDR(trainSDRIs[i])
          #print
          #tb.printSDR(testSDRIs[i])
          #junk = raw_input()
    
    # Classifier Hack, uses the testing image tags along with the SDRs from the 
    # last training cycle to interpret the SDRs from testing.
    testResults = []
    [testResults.append('') for i in range(len(testSDRIs))]
    for i,testSDRI in enumerate(testSDRIs):
      for j,trainSDRI in enumerate(trainSDRIs):
        #testSDR = np.array(tb.getSDR(testSDRI))
        #trainSDR = np.array(tb.getSDR(trainSDRI))
        #if (testSDR*trainSDR).sum() > 0:
        if testSDRI == trainSDRI:
          if len(testResults[i]) == 0:
            testResults[i] += trainingTags[j]
          elif trainingTags[j] not in testResults[i]:
            testResults[i] += "," + trainingTags[j]
    
    
    accuracy = 0.0
    recognitionMistake = False
    for i in range(len(testResults)):
      if testingTags[i] == testResults[i]:
        accuracy += 100.0/len(testResults)
      else:
        if not recognitionMistake:
          recognitionMistake = True
          print "%5s" % "Input", "Output"
        print "%-5s" % testingTags[i], testResults[i]
    
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
    
    
print "The maximum number of training cycles is set to:", maxTrainingCycles
print
print "Summary of Results"
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
    
    
