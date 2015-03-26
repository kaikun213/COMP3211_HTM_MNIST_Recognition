#!/usr/bin/env python

from obj_recog_test import object_recognition_testbench as obj_testbench
from parameter import parameter
from mnist import MNIST
import sys, getopt

def main(argv):
   p = parameter()
   try:
      opts, args = getopt.getopt(argv, "hl:t:s:d:p:",\
                   ["help", "dimension=", "precision="])
   except getopt.GetoptError:
      print 'run.py -d <nodeDimension> -p <precision>'
      print 'run.py -h for more help'
      sys.exit()
   for opt, arg in opts:
      if opt in ('-h', '--help'):
         print '-d(--dimension) <NodeDim> -p(--precision) <ConnectPrecision>'
         sys.exit()
      elif opt in ('-d', '-p' , '--dimension', '--precision'):
         p.setTestParameter(opt, arg)
   data = MNIST(p.dataPath)
   data.load_training()
   data.load_testing()
   obj_tb = obj_testbench( columnDimensions = (p.nodeDim, p.nodeDim), \
                           precision = p.precision, data = data)
   obj_tb.train(1, 100,95.0, 0)
   obj_tb.test()
   print 'NodeDimension: %i' %p.nodeDim
   print '-'*20
   print

if __name__ == "__main__":
   main(sys.argv[1:])
