# MNIST Recognition
HTM vision algorithm used for basic digit recognition of the MNIST Dataset.

- Only small course project. Report will be added later.
- Small changes to the code to work with current nupic version.
- Proven Network parameters adopted, no extra swarming done.
- Further work should integrate features for better results. *(Currently ~95% accuracy as in original vision repo)*:
  * temporal pooling,
  * sequence memory,
  * location-paired-information
  * a local receptive field


### Dependencies

Uses **NuPIC Vision Toolkit**

*"This tool kit is intended to help people who are interested in using
Numenta's NuPIC on vision problems."*

Follow the instructions for installing htmresearch, which requires
nupic and nupic.bindings (from nupic.research.core), and then use
pip to install nupic.vision.

### Install

You need to be connected to the local mysql DB *(default: root and no password, config in nupic/src/support/nupic-default.xml)* and simply run setup.py

Rest are mentioned in requirements.txt/Dependencies.md

Then follow instructions in src/nupic/vision/mnist to execute the experiments.
