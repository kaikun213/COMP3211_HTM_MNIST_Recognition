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

class SPTestBench(object):
  '''
  This class provides methods for characterizing the image recognition 
  capabilities of the spatial pooler.  The goal is to put most of the details
  in here so the top level can be as clear and concise as possible.

  The top level should have the ability to try different encoding and 
  classification schemes so there is a method for encoding with hopefully
  more to follow.  Classification is not in here yet.

  An example of how these methods might be used:

  images, tags = getImagesAndTags(trainingXMLFileName)
  vectors = imagesToVectors(images)
  for i in range(10)
    train(vectors,tags)
  images, tags = getImagesAndTags(testingXMLFileName)
  vectors = imagesToVectors(images)
  test(vectors,tags)

  '''
  def __init__(self, sp):
    '''
    The test bench has just a few things to keep track off:

    - Number of training cycles completed, just to keep the top level clean

    - Height and width of the spatial pooler's inputs and columns which are
      used for producing images of permanences and connected synapses

    - Images of permanences and connected synapses so these images do not have 
      to be generated more than necessary

    '''
    self.sp = sp

    self.trainingCyclesCompleted = 0

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
  ################################################################################
  This routine reads the XML files that contain the paths to the images and the
  tags which indicate what is in the image (i.e. "ground truth").
  ################################################################################
  '''
  def getImagesAndTags(self, XMLFileName):
    xmldoc = minidom.parse(XMLFileName)
    # Find the path to the XML file so it can be used to find the image files.
    directories = XMLFileName.split("/")
    directories.pop()
    directoryPath = ''
    for directory in directories:
      directoryPath += directory + '/'
    # Read the image list from the XML file and populate images and tags.
    imageList = xmldoc.getElementsByTagName('image') 
    images = []
    tags = []
    for image in imageList:
      tags.append(image.attributes['tag'].value)
      filename = image.attributes['file'].value
      images.append(Image.open(directoryPath + filename))
      #imagePatches[-1].show()
    return images, tags
  
  
  
  '''
  ################################################################################
  These routines convert images to bit vectors that can be used as input to
  the spatial pooler.
  ################################################################################
  '''
  def imageToVector(self, image):
    '''
    Returns a bit vector representation (list of ints) of a PIL image.
    '''
    # Convert the image to black and white
    image = image.convert('1',dither=Image.NONE)
    # Pull out the data, turn that into a list, then a numpy array,
    # then convert from 0 255 space to binary with a threshold.
    # Finally cast the values into a type CPP likes
    vector = (numpy.array(list(image.getdata())) < 100).astype('uint32')
    
    return vector
  
  
  def imagesToVectors(self, images):
    vectors = [self.imageToVector(image) for image in images]
    return vectors
  
  
  
  '''
  ################################################################################
  This routine trains the spatial pooler on the bit vectors produced from the 
  training images.  
  ################################################################################
  '''
  def train(self, trainingVectors, trainingTags):
    # Print header if this is the first training cycle
    if self.trainingCyclesCompleted == 0:
      print "\nLet the training begin!\n"
      print "%5s" % "Cycle", 
      print "%34s" % "Connected Synapse MD5 Checksum",
      print "%34s" % "Permanence MD5 Checksum",
      print "%5s" % "Cycle"
      print ""
 
    # increment cycle number and print it
    self.trainingCyclesCompleted += 1
    print "%5s" % self.trainingCyclesCompleted,

    # Get rid of old permanence and connection images
    self.permanencesImage = None
    self.connectionsImage = None

    # Return a list of all the output values
    outputValues = []

    # Initialize an array to store the column activity that results from the input.
    activeArray = numpy.zeros(self.sp.getColumnDimensions())
    
    # Feed training vectors into the spatial pooler
    for j,trainingVector in enumerate(trainingVectors):
      self.sp.compute(trainingVector, True, activeArray)
  
      # Convert activeArray to an integer
      value = 0
      for j,num in enumerate(activeArray):
        value = value << 1
        value += int(num)
      outputValues.append(value)

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
    print "%34s" % connsMD5.hexdigest(), "%34s" % permsMD5.hexdigest(),
    print "%5s" % self.trainingCyclesCompleted

    return outputValues
      
  
  
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

    # Return a list of all the output values
    outputValues = []

    # Initialize an array to store the column activity that results from the input.
    activeArray = numpy.zeros(self.sp.getColumnDimensions())
    
    # Feed training vectors into the spatial pooler
    for j,testingVector in enumerate(testingVectors):
      self.sp.compute(testingVector, True, activeArray)
  
      # Convert activeArray to an integer
      value = 0
      for j,num in enumerate(activeArray):
        value = value << 1
        value += int(num)
      outputValues.append(value)

    return outputValues
      
  
  
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
  
  
  
