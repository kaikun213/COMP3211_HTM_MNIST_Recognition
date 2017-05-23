# HTM for MNIST

These examples shows how to use an HTM network to classify images from the
MNIST dataset.

## Instructions

1. Download and extract the MNIST data:

    ```
    python -m nupic.vision.mnist.data.extract
    ```

    This will download, extract, and preprocess training and test images.
    This MNIST dataset contains 60,000 training samples and 10,000 testing
    samples. By default, images are located in a directory called `data`
    in your current working directory. Use the `--data` option to specify
    a different data directory.

2. Train and test an HTM SP-only network:

    ```
    python -m nupic.vision.mnist.run_mnist_experiment
    ```

## Results

This example achieves 95.56% accuracy on the 10,000 image training set as
of 2015-04-13.

## Acknowledgements

Numenta wishes to express its gratitude to Dr. Yann LeCun
of the Courant Institute of Mathematical Sciences,
New York, NY, for publishing the MNIST data set.
Details regarding MNIST and related publications are
available at the following URL: http://yann.lecun.com/exdb/mnist
