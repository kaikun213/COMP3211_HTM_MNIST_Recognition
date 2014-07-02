import numpy
import math
import pygame
import time
import hashlib

from PIL import Image
from pygame.color import THECOLORS
from nupic.research.spatial_pooler import SpatialPooler
from xml.dom import minidom

DEBUG = 0

class VisionTestBench(object):
  '''
  This class provides methods for characterizing nupic's image recognition 
  capabilities.  The goal is to put most of the details in here so the top 
  level can be as clear and concise as possible.
  '''
  def __init__(self, sp):
    '''
    The test bench has just a few things to keep track off:

    - Number of training cycles completed, just to keep the top level clean

    - A list of the output SDRs that is shared between the training and testing
      routines

    - Height and width of the spatial pooler's inputs and columns which are
      used for producing images of permanences and connected synapses

    - Images of permanences and connected synapses so these images do not have 
      to be generated more than necessary

    '''
    self.sp = sp

    self.trainingCyclesCompleted = 0

    self.SDRs = []

    # These images are produced together so these properties are used to allow 
    # them to be saved separately without having to generate the images twice.
    self.permanencesImage = None
    self.connectionsImage = None
    
    # Limit inputs and columns to 1D and 2D layouts for now
    # Note: only 1D seems to work with python SP
    inputDimensions = sp.getInputDimensions()
    try:
      assert(len(inputDimensions) < 3)
      self.inputHeight = inputDimensions[0]
      self.inputWidth = inputDimensions[1]
    except IndexError:
      self.inputHeight = int(numpy.sqrt(inputDimensions[0]))
      self.inputWidth = int(numpy.sqrt(inputDimensions[0]))
    except TypeError:
      self.inputHeight = int(numpy.sqrt(inputDimensions))
      self.inputWidth = int(numpy.sqrt(inputDimensions))
    
    columnDimensions = sp.getColumnDimensions()
    try:
      assert(len(columnDimensions) < 3)
      self.columnHeight = columnDimensions[0]
      self.columnWidth = columnDimensions[1]
    except IndexError:
      self.columnHeight = columnDimensions[0]
      self.columnWidth = 1
    except TypeError:
      self.columnHeight = columnDimensions
      self.columnWidth = 1
    


  '''
  ##############################################################################
  This routine trains the spatial pooler using the bit vectors produced from the 
  training images by using these vectors as input to the SP.  It continues
  training until there are no SDR collisions between input vectors that have
  different tags (ground truth) and the output SDRs are stable for 2 cycles.  
  It records the output SDRs as a list of integers that correspond to each SDR 
  and uses this list to look for SDR collisions.
  ##############################################################################
  '''
  def train(self, trainingVectors, trainingTags, maxTrainingCycles):
    # print averages of starting permanence values
    if self.trainingCyclesCompleted == 0:
      self.printPermanenceStats()
 
    # Get rid of old permanence and connection images
    self.permanencesImage = None
    self.connectionsImage = None
  
    trained = False
    LastCycleSDRNumbers = []
    while not trained and self.trainingCyclesCompleted < maxTrainingCycles:
      trained = True

      # increment cycle number 
      self.trainingCyclesCompleted += 1
  
      # Feed training vectors into the spatial pooler 
      SDRNumbers = []
      activeArray = numpy.zeros(self.sp.getColumnDimensions())
      for j,trainingVector in enumerate(trainingVectors):
        self.sp.compute(trainingVector, True, activeArray)
        # Build a list of integers corresponding to each SDR
        activeList = activeArray.tolist()
        if activeList not in self.SDRs:
          self.SDRs.append(activeList)
        SDRNumbers.append(self.SDRs.index(activeList))
    
      # check for SDR collisions
      for i in range(len(self.SDRs)):
        if SDRNumbers.count(i) > 1:
          tag = trainingTags[SDRNumbers.index(i)]
          for j in range(SDRNumbers.index(i),len(SDRNumbers)):
            if SDRNumbers[j] == i:
              if trainingTags[j] != tag:
                trained = False
  
      # check for SDR stability
      if SDRNumbers != LastCycleSDRNumbers:
        trained = False
        LastCycleSDRNumbers = SDRNumbers

      # print updated permanence stats for connected and unconnected synapses
      self.printPermanenceStats()
 
    return SDRNumbers, self.trainingCyclesCompleted
      
  
  
  '''
  ################################################################################
  This routine tests the spatial pooler on the bit vectors produced from the 
  testing images.  
  ################################################################################
  '''
  def test(self, testingVectors, testingTags):
    # Get rid of old permanence and connection images
    self.permanencesImage = None
    self.connectionsImage = None

    # Feed testing vectors into the spatial pooler 
    SDRNumbers = []
    activeArray = numpy.zeros(self.sp.getColumnDimensions())
    for j,testingVector in enumerate(testingVectors):
      self.sp.compute(testingVector, False, activeArray)
      # Build a list of integers corresponding to each SDR
      activeList = activeArray.tolist()
      if activeList not in self.SDRs:
        self.SDRs.append(activeList)
      SDRNumbers.append(self.SDRs.index(activeList))

    return SDRNumbers
      
  
  
  '''
  ################################################################################
  This routine prints the mean values of the connected and unconnected synapse
  permanences along with the percentage of synapses in each.
  It also returns the percentage of connected synapses so it can be used to 
  determine when training has finished.
  ################################################################################
  '''
  def printPermanenceStats(self):
    # Print header if this is the first training cycle
    if self.trainingCyclesCompleted == 0:
      print "\nTraining begins:\n"
      print "%5s" % "", 
      print "%17s" % "Connected", 
      print "%19s" % "Unconnected"
      print "%5s" % "Cycle", 
      print "%10s" % "Percent", 
      print "%8s" % "Mean", 
      print "%10s" % "Percent", 
      print "%8s" % "Mean"
      print ""
    pctConnected = 0
    pctUnconnected = 0
    permsMean = 0
    connectedMean = 0
    unconnectedMean = 0
    for i in range(self.columnHeight):
      perms = self.sp._permanences.getRow(i)
      numPerms = perms.size
      permsMean += perms.mean()/self.columnHeight
      connectedPerms = perms >= self.sp._synPermConnected
      numConnected = connectedPerms.sum()
      pctConnected += 100.0/self.columnHeight*numConnected/numPerms
      sumConnected = (perms*connectedPerms).sum()
      connectedMean += sumConnected/(numConnected*self.columnHeight)
      unconnectedPerms = perms < self.sp._synPermConnected
      numUnconnected = unconnectedPerms.sum()
      pctUnconnected += 100.0/self.columnHeight*numUnconnected/numPerms
      sumUnconnected = (perms*unconnectedPerms).sum()
      unconnectedMean += sumUnconnected/(numUnconnected*self.columnHeight)
    print "%5s" % self.trainingCyclesCompleted,
    print "%10s" % ("%.4f" % pctConnected),
    print "%8s" % ("%.3f" % connectedMean),
    print "%10s" % ("%.4f" % pctUnconnected),
    print "%8s" % ("%.3f" % unconnectedMean)
    return pctConnected



  '''
  ################################################################################
  This routine prints the MD5 hash of the output SDRs.
  ################################################################################
  '''
  def printOutputHash(self):
    # Print header if this is the first training cycle
    if self.trainingCyclesCompleted == 0:
      print "\nTraining begins:\n"
      print "%5s" % "Cycle", 
      print "%34s" % "Connected MD5", "%34s" % "Permanence MD5"
      print ""
    # Calculate an MD5 checksum for the permanences and connected synapses so 
    # we can see when learning has finished.
    permsMD5 = hashlib.md5()
    connsMD5 = hashlib.md5()
    for i in range(self.columnHeight):
      perms = self.sp._permanences.getRow(i)
      connectedPerms = perms >= self.sp._synPermConnected
      perms = perms.astype('string')
      [permsMD5.update(word) for word in perms]
      connectedPerms = connectedPerms.astype('string')
      [connsMD5.update(word) for word in connectedPerms]
    print "%5s" % self.trainingCyclesCompleted,
    print "%34s" % connsMD5.hexdigest(), "%34s" % permsMD5.hexdigest()



  '''
  ################################################################################
  These routines generates images of the permanences and connections of each 
  column so they can be viewed and saved.
  ################################################################################
  '''
  def calcPermsAndConns(self):
    size = (self.inputWidth*self.columnWidth,self.inputHeight*self.columnHeight)
    self.permanencesImage = Image.new('RGB', size)
    self.connectionsImage = Image.new('RGB', size)
    for j in range(self.columnWidth):
      for i in range(self.columnHeight):
        perms = self.sp._permanences.getRow(i)
        # Convert perms to RGB (effective grayscale) values
        allPerms = [(v, v, v) for v in ((1 - perms) * 255).astype('int')]
        
        connectedPerms = perms >= self.sp._synPermConnected
        connectedPerms = (numpy.invert(connectedPerms) * 255).astype('int')
        connectedPerms = [(v, v, v) for v in connectedPerms]
        
        allPermsReconstruction = self._convertToImage(allPerms, 'RGB')
        connectedReconstruction = self._convertToImage(connectedPerms, 'RGB')
        size = allPermsReconstruction.size
  
        # Add permanences and connections for each column to the images
        x = j * self.inputWidth
        y = i * self.inputHeight
        self.permanencesImage.paste(allPermsReconstruction, (x,y))
        self.connectionsImage.paste(connectedReconstruction, (x,y))


  def showPermsAndConns(self):
    if self.permanencesImage == None:
      self.calcPermsAndConns()
    size = (2*self.permanencesImage.size[0],self.permanencesImage.size[1])
    pAndCImage = Image.new('RGB', size)
    pAndCImage.paste(self.permanencesImage, (0,0))
    pAndCImage.paste(self.connectionsImage, (size[0]/2,0))
    pAndCImage.show()
        

  def showPermanences(self):
    if self.permanencesImage == None:
      self.calcPermsAndConns()
    self.permanencesImage.show()
  
  
  def showConnections(self):
    if self.connectionsImage == None:
      self.calcPermsAndConns()
    self.connectionsImage.show()
  

  def savePermsAndConns(self, filename):
    if self.permanencesImage == None:
      self.calcPermsAndConns()
    size = (2*self.permanencesImage.size[0],self.permanencesImage.size[1])
    pAndCImage = Image.new('RGB', size)
    pAndCImage.paste(self.permanencesImage, (0,0))
    pAndCImage.paste(self.connectionsImage, (size[0]/2,0))
    pAndCImage.save(filename,'JPEG')
        

  def savePermanences(self, filename):
    if self.permanencesImage == None:
      self.calcPermsAndConns()
    self.permanencesImage.save(filename,'JPEG')
        

  def saveConnections(self, filename):
    if self.connectionsImage == None:
      self.calcPermsAndConns()
    self.connectionsImage.save(filename,'JPEG')
        
  
  def _convertToImage(self, listData, mode = '1'):
    '''
    Takes in a list and returns a new square image
    '''
  
    # Assume we're getting a square image patch
    side = int(len(listData) ** 0.5)
    # Create the new image of the right size
    im = Image.new(mode, (side, side))
    # Put the data into that patch
    im.putdata(listData)
  
    return im
  
  
  
