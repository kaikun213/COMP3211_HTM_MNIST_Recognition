This tool kit is intended to help people who are interested in using Numenta's 
nupic on vision related problems.

An simple example of how these tools can be used:

<h6># get training data set </h6>
images, tags = dataset_readers.getImagesAndTags(trainingXMLFileName)
<h6># convert images to bit vectors </h6>
vectors = image_encoders.imagesToVectors(images)

<h6># train nupic on the data set for 10 repetitions </h6>
for i in range(10)
  VisionTestBench.train(vectors,tags)

<h6># get testing data set </h6>
images, tags = getImagesAndTags(testingXMLFileName)
<h6># convert images to bit vectors </h6>
vectors = imagesToVectors(images)
<h6># test nupic's image recognition accuracy </h6>
test(vectors,tags)



Look at train_and_test.py for a more detailed example.

