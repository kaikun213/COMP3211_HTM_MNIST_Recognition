# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""Wrapper around C++ extraction script."""

import argparse
import gzip
import os
import pkg_resources
import shutil
import subprocess
import tarfile
import urllib

from nupic.vision.mnist.data import convert_images

# This variable controls where we download the MNIST files from
LECUNSITE="http://yann.lecun.com/exdb/mnist/"
ARCHIVES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)



def download():
  for archive in ARCHIVES:
    if os.path.exists(archive):
      print "Archive already downloaded: {}".format(archive)
    else:
      print "Downloading {}...".format(archive),
      urllib.urlretrieve(LECUNSITE + archive, archive)
      print " done"



def extract():
  for archive in ARCHIVES:
    outputName = archive.split(".")[0]
    if os.path.exists(outputName):
      print "Already extracted: {}".format(outputName)
    else:
      print "Extracting {}...".format(archive),
      with gzip.open(archive, "rb") as fIn, open(outputName, "wb") as fOut:
        shutil.copyfileobj(fIn, fOut)
      print " done"



def preprocess():
  print "Preprocessing MNIST data...",
  executable = pkg_resources.resource_filename(
      "nupic.vision.mnist.data", "extract_mnist")
  subprocess.check_call(executable, shell=True)
  print " done"



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataDir", default="data",
                      help="location to put data (default: ./data)")
  args = parser.parse_args()
  try:
    os.makedirs(args.dataDir)
  except os.error:
    pass
  os.chdir(args.dataDir)
  download()
  extract()
  preprocess()
  convert_images.doConversion(os.getcwd())
