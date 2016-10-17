# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

"""Example showing usage of PCANode."""

from nupic.engine import Network

from nupic.vision.regions.ImageSensor import ImageSensor
from nupic.vision.regions.PCANode import PCANode



def runExperiment():
  Network.unregisterRegion("ImageSensor")
  Network.registerRegion(ImageSensor)
  Network.registerRegion(PCANode)
  inputSize = 8

  net = Network()
  sensor = net.addRegion(
      "sensor", "py.ImageSensor" ,
      "{ width: %d, height: %d }" % (inputSize, inputSize))

  params = ("{bottomUpCount: %s, "
            " SVDSampleCount: 5, "
            " SVDDimCount: 2}" % inputSize)

  pca = net.addRegion("pca", "py.PCANode", params)

  linkParams = "{ mapping: in, rfSize: [%d, %d] }" % (inputSize, inputSize)
  net.link("sensor", "pca", "UniformLink", linkParams, "dataOut", "bottomUpIn")

  net.initialize()

  for i in range(10):
    pca.getSelf()._testInputs = numpy.random.random([inputSize])
    net.run(1)
    print s.sendRequest("nodeOPrint pca_node")



if __name__=="__main__":
  runExperiment()
