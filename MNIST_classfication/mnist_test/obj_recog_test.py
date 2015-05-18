#usr/bin/env python

import os
import Image
import numpy as np
from obj_recog import Object_Recognition as obj_sub
from nupic.algorithms.KNNClassifier import KNNClassifier
class object_recognition_testbench():
   
   def __init__(self,
                inputShape = (28, 28),
                columnDimensions = (16, 16),
                activeRate = 0.1, precision = 8, data = 0):
      self.obj_module = obj_sub(inputShape, columnDimensions, \
                                activeRate, precision)
      self.clf = KNNClassifier()
      self.SDRList = []
      self.testResultTag = []   
      self.trainImagesVectors = (np.array(list(data.train_images))>128)\
      .astype('uint32')
      self.testImagesVectors = (np.array(list(data.test_images))>128)\
      .astype('uint32')
      self.trainTags = data.train_labels
      self.testTags = data.test_labels
      self.buildTagsTable(self.trainTags, 1)
   def buildTagsTable(self, tags, verbosity = 0): 
      self.tagsTable = []
      for tag in tags:
         if tag not in self.tagsTable:
            self.tagsTable.append(tag)
      if verbosity:
         print self.tagsTable
   def learnImages(self, verbosity=0):
      #activeArray = np.zeros(self.obj_module.sp.getNumiColumns());
      total = len(self.trainImagesVectors)
      print 'SP learning phase'
      for j,trainImage in enumerate(self.trainImagesVectors):
         self.obj_module.spatialPooling(trainImage, True)
         if j % (total/10) == 0:
            print 1.0*j/total*100.0
            #print trainImage         
      print '-'*30
      print
      print 'Classifier learning phase'
      for j,trainImage in enumerate(self.trainImagesVectors):
         if j % (total/10) == 0:
            print 1.0*j/total*100.0
         category = self.tagsTable.index(self.trainTags[j])
         activeArray = self.obj_module.spatialPooling(trainImage, False)
         self.clf.learn(activeArray, category)
         #if j % 1000 == 0:   
         #   print activeArray
         #   print category
         if verbosity:
            print 'Train Image : %r' % trainImage
            print 'ActiveList: %r' % activeList
      print '-'*30
      print
   def checkAccuracy(self, datas, tags, verbosity=0):
      total = len(datas)
      accuracy = 0.0
      print 'Testing phase'
      #f = open('result.txt', 'w')
      #f.write('inferred\tcategory')
      if verbosity:
         print 'Check Accuracy, len = %s' % len(tags)
         print tags
      for j,data in enumerate(datas):
         if j % (total/10) == 0:
            print 1.0*j/total*100.0
         category = self.tagsTable.index(tags[j])
         activeArray = self.obj_module.spatialPooling(data, False)
         inferred = self.clf.getClosest(activeArray)[0]
         #if j % 1000 == 0:
         #   print activeArray
         #   print category
         #   print inferred
         #f.write(str(inferred))
         #f.write('\t')
         #f.write(str(category))
         #f.write('\n')
         if inferred == category:
            accuracy += 100.0/total
         if verbosity:
            print 'Pattern i = %s , answer = %s, inferred = %s' \
                  %(j, category, inferred) 
            print 'Check Result: %r' % (inferred==category)
      print '-'*30
      print
      #f.close()
      return accuracy

   def test(self):
      print '*'*30
      print 'Use test data to test SP'
      accuracy = self.checkAccuracy(self.testImagesVectors, self.testTags)
      #print self.testTags
      print 'testing accuracy: %s' % accuracy
   
   def train(self, minIteration=10, maxIteration=100, \
             minAccuracy=95.0, verbosity=0):
      accuracy = 0.0
      iteration = 0
      stop_threshold = 1.0
      with open('iteration.txt','w') as f:
         while (minAccuracy > accuracy and \
         abs(minAccuracy - accuracy) > stop_threshold and  \
         iteration < maxIteration) or iteration < minIteration:
            iteration += 1
            self.clf.clear()
            self.learnImages()
            accuracy = self.checkAccuracy(self.trainImagesVectors, \
                                       self.trainTags)
            f.write(str(iteration))
            f.write('\t')
            f.write(str(accuracy))
            f.write('\n')
            print 'training accuracy: %s' % accuracy
            if verbosity:
               print 'Iteration = %s , Accuracy = %s ' % (iteration ,accuracy)

   def saveTestResult(self, mode, saveRoot, verbosity = 1):
      for tag in self.tagsTable:
         saveDir = saveRoot + '/' + tag
         if not os.path.exists(saveDir):
            os.mkdir(saveDir) 

      for j, image in enumerate(self.testImages):
         if mode in ('p', 'png'):
            fileExt = 'png'
            saveDir = saveRoot + '/' + self.testResultTag[j] 
            if not os.path.exists(saveDir):
              print 'Error: This tag %s is not included in tagsTable!' \
                     %(self.testResultTag[j])
            savePath = saveDir + '/' + \
                       self.testFilenames[j].split('/')[1].split('.')[0] +\
                       '.' + fileExt
            image.save(savePath)
         #elif mode in ('x', 'xml'):
