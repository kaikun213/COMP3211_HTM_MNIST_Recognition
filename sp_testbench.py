import numpy
import math
import pygame
import time

from PIL import Image
from pygame.color import THECOLORS
from copy import copy
from nupic.research.spatial_pooler import SpatialPooler
from xml.dom import minidom

DEBUG = 0

class SPTestBench(object):
  '''
  This class provides methods for characterizing the image recognition 
  capabilities of the spatial pooler.
  '''
  def __init__(self, sp):
    '''
    Pass a reference to the spatial pooler once so we don't have to do it 
    over and over.
    '''
    self.sp = sp

    # Limit inputs and columns to 1D and 2D layouts for now
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
    
    self.permanencesImage = None
    self.connectionsImage = None
    

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
  def train(self, trainingVectors):
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
  
  
  
