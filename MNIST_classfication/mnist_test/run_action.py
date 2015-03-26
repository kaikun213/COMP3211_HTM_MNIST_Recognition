#!/usr/bin/env python

from obj_recog_test import object_recognition_testbench as obj_testbench
from parameter import parameter
import sys, getopt

def main(argv):
   p = parameter()
   try:
      opts, args = getopt.getopt(argv, "hl:t:s:",\
                   ["help", "learn=", "test=", "saveMode="])
   except getopt.GetoptError:
      print 'run.py -l <trainSet> -t <testSet>'
      print 'run.py -h for more help'
      sys.exit()
   saveEnable = False
   for opt, arg in opts:
      if opt in ('-h', '--help'):
         print '-l(--learn) <trainSet> -t(--test) <testSet>'
         sys.exit()
      #elif opt in ('-l', '--learn'):
      #   self.trainingDataset = self.dataPath + dataset
      #   print 'Set Train Set : %s' % dataset
      #elif opt in ('-t', '--test'):
      #   self.testingDataset = self.dataPath + dataset
      elif opt in ('-s', '--saveMode'):
         saveEnable = True
         saveMode = arg
      else:
         p.setSet(opt, arg)
   obj_tb = obj_testbench()
   obj_tb.changeTrainingImg(p.trainingDataset)
   obj_tb.changeTestingImg(p.testingDataset)
   obj_tb.train(20,95.0, 1)
   obj_tb.test()
   if saveEnable:
      obj_tb.saveTestResult(saveMode, p.savePath)

if __name__ == "__main__":
   main(sys.argv[1:])
