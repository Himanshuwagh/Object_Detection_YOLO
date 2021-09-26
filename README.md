## Introduction

This is the code for [this](https://youtu.be/4eIBisqx9_g) video on Youtube by Siraj Raval

[![Build Status](https://travis-ci.org/thtrieu/darkflow.svg?branch=master)](https://travis-ci.org/thtrieu/darkflow) [![codecov](https://codecov.io/gh/thtrieu/darkflow/branch/master/graph/badge.svg)](https://codecov.io/gh/thtrieu/darkflow)

Real-time object detection and classification. Paper: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).

Read more about YOLO (in darknet) and download weight files [here](http://pjreddie.com/darknet/yolo/). In case the weight file cannot be found, I uploaded some of mine [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU), which include `yolo-full` and `yolo-tiny` of v1.0, `tiny-yolo-v1.1` of v1.1 and `yolo`, `tiny-yolo-voc` of v2.

### What is YOLO?
- YOLO takes a completely different approach.
- It’s not a traditional classifier that is repurposed to be an object detector.
- YOLO actually looks at the image just once (hence its name: You Only Look Once) but in a clever way.

YOLO divides up the image into a grid of 13 by 13 cells:

<img src="https://camo.githubusercontent.com/4338301d905b87a1e9d8a4b68b63775d30adcf15/687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f477269644032782e706e67">

- Each of these cells is responsible for predicting 5 bounding boxes.
- A bounding box describes the rectangle that encloses an object.
- YOLO also outputs a confidence score that tells us how certain it is that the predicted bounding box actually encloses some object.
- This score doesn’t say anything about what kind of object is in the box, just if the shape of the box is any good.
The predicted bounding boxes may look something like the following (the higher the confidence score, the fatter the box is drawn):

<img src="https://camo.githubusercontent.com/c5b0bb9269257ba704824dedd8bc03b62d40ed61/687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f426f7865734032782e706e67">

- For each bounding box, the cell also predicts a class.
- This works just like a classifier: it gives a probability distribution over all the possible classes.
- YOLO was trained on the PASCAL VOC dataset, which can detect 20 different classes such as:

             - bicycle
             - boat
             - car
             - cat
             - dog
             - person

- The confidence score for the bounding box and the class prediction are combined into one final score that tells us the probability that this bounding box contains a specific type of object.

- For example, the big fat yellow box on the left is 85% sure it contains the object “dog”:

<img src="https://camo.githubusercontent.com/d1becdbc2064341b828ae5a73d6f92f4a8a27308/687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f53636f7265734032782e706e67">

- Since there are 13×13 = 169 grid cells and each cell predicts 5 bounding boxes, we end up with 845 bounding boxes in total.
- It turns out that most of these boxes will have very low confidence scores, so we only keep the boxes whose final score is 30% or more (you can change this threshold depending on how accurate you want the detector to be).
- The final prediction is then:
<img src="https://camo.githubusercontent.com/08d5da9cb0436313a14c81b99f370d43fba01f98/687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f50726564696374696f6e4032782e706e67">

- From the 845 total bounding boxes we only kept these three because they gave the best results.
- But note that even though there were 845 separate predictions, they were all made at the same time — the neural network just ran once. And that’s why YOLO is so powerful and fast.

### The architecture of YOLO is simple, it’s just a convolutional neural network:

<img src="https://camo.githubusercontent.com/3c2151338f97e8494cb208d46a29bab4763c7dd6/68747470733a2f2f692e696d6775722e636f6d2f5148304376524e2e706e67">

This neural network only uses standard layer types: convolution with a 3×3 kernel and max-pooling with a 2×2 kernel. No fancy stuff. There is no fully-connected layer in YOLOv2.

The very last convolutional layer has a 1×1 kernel and exists to reduce the data to the shape 13×13×125. This 13×13 should look familiar: that is the size of the grid that the image gets divided into.

So we end up with 125 channels for every grid cell. These 125 numbers contain the data for the bounding boxes and the class predictions. Why 125? Well, each grid cell predicts 5 bounding boxes and a bounding box is described by 25 data elements:

- x, y, width, height for the bounding box’s rectangle
- the confidence score
- the probability distribution over the classes
Using YOLO is simple: you give it an input image (resized to 416×416 pixels), it goes through the convolutional network in a single pass, and comes out the other end as a 13×13×125 tensor describing the bounding boxes for the grid cells. All you need to do then is compute the final scores for the bounding boxes and throw away the ones scoring lower than 30%.

### Improvements to YOLO v1
### YoLO v2 vs YoLO v1

Speed (45 frames per second — better than realtime)
Network understands generalized object representation (This allowed them to train the network on real world images and predictions on artwork was still fairly accurate).
faster version (with smaller architecture) — 155 frames per sec but is less accurate.
Paper here https://arxiv.org/pdf/1612.08242v1.pdf

## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.
