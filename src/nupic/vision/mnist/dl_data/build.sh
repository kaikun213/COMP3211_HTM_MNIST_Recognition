#!/bin/bash
g++ -c extract_mnist.cpp
g++ -o extract_mnist extract_mnist.o
rm extract_mnist.o
