This tool kit is intended to help people who are interested in using Numenta's 
nupic on vision related problems.

An simple example of how these tools can be used:

<h3># get training data set </h3>
images, tags = dataset_readers.getImagesAndTags(trainingXMLFileName)
<h3># convert images to bit vectors </h3>
vectors = image_encoders.imagesToVectors(images)

<h3># train nupic on the data set for 10 repetitions </h3>
for i in range(10)
  VisionTestBench.train(vectors,tags)

<h3># get testing data set </h3>
images, tags = getImagesAndTags(testingXMLFileName)
<h3># convert images to bit vectors </h3>
vectors = imagesToVectors(images)
<h3># test nupic's image recognition accuracy </h3>
test(vectors,tags)



Look at train_and_test.py for a more detailed example.

