# 3D-AlexNet-Tensorflow-Tensorboard
Implementation of AlexNet for 3D images in Tensorflow

AlexNet is the CNN designed by Alex Krizhevsky for the "ImageNet Large Scale Visual Recognition Challenge". The Network had a very similar architecture to LeNet (developed by Yann LeCun in 1990’s), but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer).

Alexnet contained only 8 layers, first 5 were convulational layers followed by fully connected layers. Here I used the same structure, but modified to fit 3D images.

## Training 
```
python MainCode.py 
```

## Reference paper: 
http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

## Slides on AlexNet
http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf
