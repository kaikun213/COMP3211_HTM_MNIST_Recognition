# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

from copy import deepcopy

from nupicvision.regions.ImageSensorExplorers.BaseExplorer import BaseExplorer


_DIRECTIONS = ("left", "right", "up", "down")


class RandomSaccade(BaseExplorer):
  """ Explorer that implements basic random saccades (movements
  around an image)
  """


  def __init__(self, replacement=True, saccadeMin=10, saccadeMax=10,
               numSaccades=8, maxDrift=0,
               *args, **kwargs):
    """
    :param bool replacement: Allow the explorer to repeat images (if true)
    :param int saccadeMin: Minimum distance a saccade will travel (in px)
    :param int saccadeMax: Maxium distance a saccade will travel (in px)
    :param int numSaccades: Number of saccades to run over each image
    :param maxDrift: Amount (in px) a saccade's center (fovea) can move out
      of the bounds of the image
    """
    BaseExplorer.__init__(self, *args, **kwargs)

    self.replacement = replacement
    self.saccadeMin = saccadeMin
    self.saccadeMax = saccadeMax
    self.numSaccades = numSaccades
    self.maxDrift = maxDrift

    self.prevSaccade = None

    if not self.replacement:
      self.history = []
      self.imageHistory = []

    self.saccadeIndex = 0


  def seek(self, iteration=None, position=None):
    """
    Seek to the specified position or iteration.

    iteration -- Target iteration number (or None).
    position -- Target position (or None).

    ImageSensor checks validity of inputs, checks that one (but not both) of
    position and iteration are None, and checks that if position is not None,
    at least one of its values is not None.

    Updates value of position.
    """

    # Zero out the history when seeking to iteration 0. This so we can replicate
    #  how random explorers behave in the vision framework and NVT.
    if iteration is not None and iteration == 0:
      if not self.replacement:
        self.history = []
    BaseExplorer.seek(self, iteration=iteration, position=position)


  def first(self, seeking=False):
    "Just used in initialization for ImageSensor..."
    if not self.numImages:
      return

    self.history = []
    self.imageHistory = []

    BaseExplorer.first(self)
    self.prevSaccade = None
    self.position["image"] = self.pickRandomImage(self.random)
    self.saccadeIndex = 0

  def next(self, seeking=False):
    """ Move to the next saccade position if number of saccades on an image
    has not reached it's maximum. Otherwise, load the next random image.
    """
    if not self.numImages:
      return

    if not self.replacement \
        and len(self.history) == self.getNumIterations(None):
      # All images have been visited
      self.history = []
      self.imageHistory = []

    while True:
      if self.saccadeIndex is 0 or self.saccadeIndex > (self.numSaccades - 1):
        while self.position["image"] in self.imageHistory:
          BaseExplorer.first(self)
          self.position["image"] = self.pickRandomImage(self.random)
        self.imageHistory.append(self.position["image"])
        self.prevSaccade = {"prevOffset": deepcopy(self.position["offset"]),
                            "direction": None,
                            "length": None,
                            "newOffset": deepcopy(self.position["offset"])}
        historyItem = (self.position["image"],
                       self.saccadeIndex)
        if not self.replacement:
          # Add to the history
          self.history.append(historyItem)
        self.saccadeIndex = 1
        return

      if not seeking:
        saccadeDirection = self.random.choice(_DIRECTIONS)
        saccadeLength =  self.random.randint(self.saccadeMin, self.saccadeMax)

      historyItem = (self.position["image"],
                     self.saccadeIndex)
      if self.replacement or historyItem not in self.history:
        # Use this position
        if not seeking:
          imageSize = self.getFilteredImages()[0].size
          if not self._checkFoveaInImage(imageSize,
                                         saccadeLength,
                                         saccadeDirection):
            continue

          self.prevSaccade = {"prevOffset": deepcopy(self.position["offset"]),
                              "direction": saccadeDirection,
                              "length": saccadeLength}
          if saccadeDirection == "left":
            self.position["offset"][0] -= saccadeLength
          elif saccadeDirection == "right":
            self.position["offset"][0] += saccadeLength
          elif saccadeDirection == "up":
            self.position["offset"][1] -= saccadeLength
          elif saccadeDirection == "down":
            self.position["offset"][1] += saccadeLength
          self.prevSaccade["newOffset"] = deepcopy(self.position["offset"])


        if not self.replacement:
          # Add to the history
          self.history.append(historyItem)
        self.saccadeIndex += 1
        return

  def _checkFoveaInImage(self, imageSize, saccadeLength, saccadeDirection):
    """ Used to check that a saccade falls within the bounds of the image
    (within the amount of drift specified during initialization)
    """
    if saccadeDirection == "left":
      foveaCenter = [(((imageSize[0] - self.enabledWidth) / 2) +
                      self.position["offset"][0] - saccadeLength +
                      (self.enabledWidth / 2)),
                     (((imageSize[1] - self.enabledHeight) / 2) +
                      self.position["offset"][1] +
                      (self.enabledHeight / 2))]
    elif saccadeDirection == "right":
      foveaCenter = [(((imageSize[0] - self.enabledWidth) / 2) +
                      self.position["offset"][0] + saccadeLength +
                      (self.enabledWidth / 2)),
                     (((imageSize[1] - self.enabledHeight) / 2) +
                      self.position["offset"][1] +
                      (self.enabledHeight / 2))]
    elif saccadeDirection == "up":
      foveaCenter = [(((imageSize[0] - self.enabledWidth) / 2) +
                      self.position["offset"][0] +
                      (self.enabledWidth / 2)),
                     (((imageSize[1] - self.enabledHeight) / 2) +
                      self.position["offset"][1] - saccadeLength +
                      (self.enabledHeight / 2))]
    elif saccadeDirection == "down":
      foveaCenter = [(((imageSize[0] - self.enabledWidth) / 2) +
                      self.position["offset"][0] +
                      (self.enabledWidth / 2)),
                     (((imageSize[1] - self.enabledHeight) / 2) +
                      self.position["offset"][1] + saccadeLength +
                      (self.enabledHeight / 2))]
    else:
      raise RuntimeError("saccadeDirection did not match one of the following:"
                         "[left, right, up, down]")
    if ((foveaCenter[0] not in
         range(0 - self.maxDrift, imageSize[0] + self.maxDrift)) or
        (foveaCenter[1] not in
         range(0 - self.maxDrift, imageSize[1] + self.maxDrift))):
      return False

    return True


  def getNumIterations(self, image):
    """
    Get the number of iterations required to completely explore the input space.

    image -- If None, returns the sum of the iterations for all the
      loaded images. Otherwise, image should be an integer specifying the
      image for which to calculate iterations.

    ImageSensor takes care of the input validation.
    """

    if self.replacement:
      raise RuntimeError("RandomEyeMovements only supports getNumIterations() "
                         "when 'replacement' is False.")
    else:
      if image is not None:
        return self.numSaccades
      else:
        return self.numSaccades * self.numImages