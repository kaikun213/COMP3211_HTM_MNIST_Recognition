#!/usr/bin/env python
# It's a test program for image read in python

import os,sys
import Image
import numpy as np
from nupic.research.TP10X2 import TP10X2 as TP

print len(sys.argv)
png = Image.open(sys.argv[1]).convert('LA')
png.save('test.png')
print png.size
png_arr = np.asarray(png.getdata()).reshape(png.size[1], png.size[0], -1)
print len(png_arr)
print len(png_arr[1])


#for row in png_arr:
#   print len(row)
#   for pixel in row:
#      print len(pixel)      
#      if pixel != 255:
#         print pixel
   #print len(pixel)
   #print pixeli
