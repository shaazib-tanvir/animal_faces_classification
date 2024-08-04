Animal Faces Classification
============

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## What is this?
A Convolutional Neural Network (CNN) made in PyTorch to classify images as a cat, dog or some other wild animal.

![Example](https://i.imgur.com/9yelY1h.png)

CC BY-NC 4.0 

## Dataset

The dataset used to train this model is <https://www.kaggle.com/datasets/andrewmvd/animal-faces>.
Author: Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha

## Setup & Usage
Clone this repo and install pytorch and pandas with `pip` in a virtual environment.
The model can be used without the dataset but to train it you'll need to put the dataset in a directory in named `data/`.
For more details visit: <https://pytorch.org/get-started/locally/>

Run `python src/main.py` to train the model and `python src/example_prediction.py` to make a prediction.
