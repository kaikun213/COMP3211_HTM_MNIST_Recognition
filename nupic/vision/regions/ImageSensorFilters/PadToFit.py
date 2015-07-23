# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

"""
## @file
"""

from PIL import Image

from nupic.vision.regions.ImageSensorFilters.BaseFilter import BaseFilter


class PadToFit(BaseFilter):

  """
  ** DEPRECATED ** Pad the image so that it fits the specified size.
  """

  def __init__(self, width, height):
    """
    ** DEPRECATED **
    @param width -- Target width, in pixels.
    @param height -- Target height, in pixels.
    """

    BaseFilter.__init__(self)

    self.width = width
    self.height = height

  def process(self, image):
    """
    @param image -- The image to process.

    Returns a single image, or a list containing one or more images.
    """

    BaseFilter.process(self, image)

    if image.size == (self.width, self.height):
      return image

    if image.size[0] > self.width or image.size[1] > self.height:
      raise RuntimeError('Image is larger than target size')

    newImage = Image.new(image.mode, (self.width, self.height),
      self.background)
    xPad = self.width - image.size[0]
    yPad = self.height - image.size[1]
    newImage.paste(image, (xPad/2, yPad/2))
    return newImage
