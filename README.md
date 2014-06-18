This tool kit is intended to help people who are interested in using Numenta's 
nupic on vision related problems.

An simple example of how these tools can be used:

<b># get training data set </b>
images, tags = dataset_readers.getImagesAndTags(trainingXMLFileName)
<b># convert images to bit vectors </b>
vectors = image_encoders.imagesToVectors(images)

<b># train nupic on the data set for 10 repetitions </b>
for i in range(10)
  VisionTestBench.train(vectors,tags)

<b># get testing data set </b>
images, tags = getImagesAndTags(testingXMLFileName)
<b># convert images to bit vectors </b>
vectors = imagesToVectors(images)
<b># test nupic's image recognition accuracy </b>
test(vectors,tags)



Look at train_and_test.py for a more detailed example.

