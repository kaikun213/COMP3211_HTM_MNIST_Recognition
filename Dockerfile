# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

FROM ubuntu:16.04

ENV HOME=/root
ENV PATH=$HOME/.local/bin:$PATH
ENV PYTHONPATH=$HOME/.local/lib/python2.7/site-packages
RUN echo $PYTHONPATH

RUN apt-get update
RUN apt-get install -y build-essential python python-dev cmake python-pip git-core pkg-config libfreetype6-dev libpng-dev zlib1g-dev curl

RUN pip install --user -U setuptools
RUN pip install --user -U pip
RUN pip install --user -U wheel

RUN mkdir $HOME/nta
WORKDIR $HOME/nta

# Clone relevant repos
RUN git clone https://github.com/numenta/nupic.research.core
RUN git clone https://github.com/numenta/nupic
RUN git clone https://github.com/numenta/nupic.research
RUN git clone https://github.com/numenta/nupic.vision

# Build & install nupic.bindings
WORKDIR $HOME/nta/nupic.research.core
RUN pip install --user numpy==1.11.2 pycapnp==0.5.8
RUN mkdir -p build/scripts
WORKDIR $HOME/nta/nupic.research.core/build/scripts
RUN cmake ../.. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../release -DNUPIC_IWYU=OFF -DPY_EXTENSIONS_DIR=../../bindings/py/nupic/bindings
RUN make -j4 && make install
WORKDIR $HOME/nta/nupic.research.core
RUN pip install --user -e .

# Install nupic
WORKDIR $HOME/nta/nupic
RUN pip install --user -e .

# Install nupic.research
WORKDIR $HOME/nta/nupic.research
RUN pip install --user -e .

# Install nupic.vision
WORKDIR $HOME/nta/nupic.vision
# We use --no-deps because it complains about nupic.research - not sure why
# but perhaps it doesn't understand the difference in package and namespace
# names
RUN pip install --user pillow enum34
RUN pip install --user --no-deps -e .

# Set up MNIST data
WORKDIR $HOME/nta/nupic.vision/src/nupic/vision/mnist
RUN /bin/bash extract_mnist.sh
RUN mkdir mnist
RUN mv mnist_extraction_source/training mnist/
RUN mv mnist_extraction_source/testing mnist/
RUN python ./convertImages.py mnist
RUN /bin/bash create_training_sample.sh

# Run the experiment
WORKDIR $HOME/nta/nupic.vision/src/nupic/vision/mnist
RUN ls -l
RUN python saccades_experiment.py
