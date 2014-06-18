This tool kit is intended to help people who are interested in using Numenta's 
nupic on vision related problems.

An simple example of how these tools can be used:

```
# get training data set 
images, tags = dataset_readers.getImagesAndTags(trainingXMLFileName)
# convert images to bit vectors 
vectors = image_encoders.imagesToVectors(images)

# train nupic on the data set for 10 repetitions 
for i in range(10)
  VisionTestBench.train(vectors,tags)

# get testing data set 
images, tags = dataset_readers.getImagesAndTags(testingXMLFileName)
# convert images to bit vectors 
vectors = image_encoders.imagesToVectors(images)
# test nupic's image recognition accuracy 
VisionTestBench.test(vectors,tags)
```


Look at train_and_test.py for a more detailed example.

