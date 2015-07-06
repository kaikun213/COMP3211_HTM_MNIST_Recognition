# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2006-2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os
import sys

import numpy
from PIL import Image

def doConversion(dataDir):

  # Inner function to process each directory
  def _visitorProc(state, dirname, names):

    # Inner function to convert one image
    def _convertData(srcPath, dstPath):
      fpSrc = open(srcPath, 'r')
      rawLines = fpSrc.readlines()
      numRows, numCols = [int(token) for token in rawLines[0].split()]
      pixelRows = rawLines[1:]
      pixels = []
      for pixelRow in pixelRows:
        pixels += [int(token) for token in pixelRow.strip().split()]
      fpSrc.close()
      # Create array
      numpyImg = numpy.array(pixels, dtype=numpy.uint8).reshape(numRows, numCols)
      image = Image.fromarray(numpyImg, "L")
      image.save(dstPath)
      # Destroy original text version
      os.remove(srcPath)

    # Process the contents of the directory
    for name in names:
      imgName, imgExt = os.path.splitext(name)
      if imgExt == '.txt':
        srcPath = os.path.join(dirname, name)
        dstPath = os.path.join(dirname, imgName + ".png")
        print "%s ==> %s" % (srcPath, dstPath)
        _convertData(srcPath, dstPath)
        state['numImages'] += 1

  # Perform final conversion
  state = dict(numImages=0)
  os.path.walk(os.path.join(dataDir, "training"), _visitorProc, state)
  os.path.walk(os.path.join(dataDir, "testing"), _visitorProc, state)
  print "Total images: %d" % state['numImages']


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print "Usage: python convertImages.py dataDir"
  dataDir = os.path.join(os.getcwd(), sys.argv[1])
  doConversion(dataDir)
