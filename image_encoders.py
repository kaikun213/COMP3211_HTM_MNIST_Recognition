import numpy

from PIL import Image


'''
################################################################################
These routines convert images to bit vectors that can be used as input to
the spatial pooler.
################################################################################
'''
def imageToVector(image):
  '''
  Returns a bit vector representation (list of ints) of a PIL image.
  '''
  # Convert the image to black and white
  image = image.convert('1',dither=Image.NONE)
  # Pull out the data, turn that into a list, then a numpy array,
  # then convert from 0 255 space to binary with a threshold.
  # Finally cast the values into a type CPP likes
  vector = (numpy.array(list(image.getdata())) < 100).astype('uint32')

  return vector


def imagesToVectors(images):
  vectors = [imageToVector(image) for image in images]
  return vectors



