## AI-Powered Image Classifier – MNIST

A PyTorch-based Convolutional Neural Network (CNN) implementation for handwritten digit recognition using the MNIST dataset. This project demonstrates a deep learning pipeline from data loading and preprocessing to model training and evaluation, achieving around 98% accuracy on the test set.

## Features

-End-to-end CNN model for image classification

-Training and validation on the popular MNIST dataset

-Supports GPU acceleration for faster training (if available)

-Clear and modular PyTorch codebase

-Easily extensible to other image classification tasks

## Installation

git clone https://github.com/Adish7Pandya/ai-mnist-classifier.git
cd ai-mnist-classifier
pip install -r requirements.txt

##Usage

-Install dependencies:

pip install -r requirements.txt

-Train the model:

python main.py

The script will:

-Automatically download the MNIST dataset if not present

-Train the CNN model for a predefined number of epochs

-Evaluate the model on the test set

-Print training and test accuracy metrics

-Optional: GPU Training
If you have a CUDA-enabled GPU, PyTorch will automatically use it to speed up training.

## Result

-Achieves approximately 98% test accuracy

-Training time depends on hardware but is optimized for efficiency

## Future Work

-Add support for hyperparameter tuning

-Extend to other image datasets (e.g., CIFAR-10)

-Integrate TensorBoard for training visualization

-Deploy the model with a web interface

## Author 

-Adish Pandya
Feel free to reach out for questions or collaboration!

