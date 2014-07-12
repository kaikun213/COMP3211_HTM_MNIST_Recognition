'''
This module contains different classifiers to try when using NuPIC on visual
tasks.
'''

'''
  Methods available in KNNClassifier

  def clear(self):
  def prototypeSetCategory(self, idToRelabel, newCategory):
  def removeIds(self, idsToRemove):
  def removeCategory(self, categoryToRemove):
  def doIteration(self):
  def learn(self, inputPattern, inputCategory, partitionId=None, isSparse=0,
              rowID=None):
  def getOverlaps(self, inputPattern):
  def getDistances(self, inputPattern):
  def infer(self, inputPattern, computeScores=True,
                  overCategories=True, partitionId=None):
  def getClosest(self, inputPattern, topKCategories = 3):
  def closestTrainingPattern(self, inputPattern, cat):
  def closestOtherTrainingPattern(self, inputPattern, cat):
  def getPattern(self, idx, sparseBinaryForm=False, cat=None):
  def finishLearning(self):
  def restartLearning(self):
  def computeSVD(self, numSVDSamples=None, finalize=True):
  def getAdaptiveSVDDims(self, singularValues, fractionOfMax=0.001):
  def finalizeSVD(self, numSVDDims=None):
  def leaveOneOutTest(self):
  def remapCategories(self, mapping):
  def remapCategories(self, mapping):
  def setCategoryOfVectors(self, vectorIndices, categoryIndices):
'''
from nupic.algorithms.KNNClassifier import KNNClassifier


'''
  Methods available in exactMatch

  def clear(self):
  def learn(self, inputPattern, inputCategory, isSparse=0):
  def infer(self, inputPattern):
'''
class exactMatch(object):
  '''
  This classifier builds a list of SDRs and their associated categories.  When
  queried for the category of an SDR it returns the first category in the list
  that has a matching SDR.
  '''
  def __init__(self):
    '''
    This classifier has just two things to keep track off:

    - A list of the known categories 

    - A list of the SDRs associated with each category
    '''
    self.SDRs = []
    self.categories = []


  def clear(self):
    self.SDRs = []
    self.categories = []


  def learn(self, inputPattern, inputCategory, isSparse=0):
    inputList = inputPattern.astype('int32').tolist()
    if inputList not in self.SDRs:
      self.SDRs.append(inputList)
      self.categories.append([inputCategory])
    else:
      self.categories[self.SDRs.index(inputList)].append(inputCategory)


  def infer(self, inputPattern):
    inputList = inputPattern.astype('int32').tolist()
    if inputList in self.SDRs:
      winner = self.categories[self.SDRs.index(inputList)][0]
      # format return value to match KNNClassifier
      result = (winner, [], [], [])
      return result
  

