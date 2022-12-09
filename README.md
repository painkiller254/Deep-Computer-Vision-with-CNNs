# Deep-Computer-Vision-with-CNNs
## Convolution Neural Networks

## Convolution Layer
The most important building block of a CNN is the convolution layer. Neaurons in the firt convolution layer are not connected to every single pixel in the input image, but only to pixels in their receptive fields.

In turn, each neuron in the second convolution layer is connected only to neurons loated within a small rectangle in the first layer.

This arhitecture allows the network to concentrate on small low-lvel features in the first hidden layer, then assemble them into larger higher-level features in the next hidden layer, asn so on,


## Filters

A neurons weights can be represented as a small image the size of the receptive field. 
A layer full of neurons using the same filter outputs a feature map, which highlights the areas in an image that actiavte the filter the most.

Of course, ypu dont have to define the filters manually: instead, during training the convolution layer will automatically learn the most useful filters for its task, and the layers above will learn to combine them into more complex patterns.


## Stacking Multiple Feature Maps

A convolution layer has multiple filters, (you decide how many), and it outputs one feature map per filter, so it is more accurately represented in 3D.

TO do so, it has one neuron per pixel in each feature map, and all the neurons within a given feature map share the same parameters.

However, neurons in different feature maos use different parameters. 

A neuron's receptive field is the same as described earlier, but it extends across all the previous layers feature maps.

In short a convolution layer simultaneously applies multiple trainable filters to its inputs, making it capable of detecting multiple features anywhere in its inputs.

The fact that all neurons in a feature map share the same parameters dramtically reduces the number of parameters in the model.

Moreover, once the CNN has learned to recognize a pattern in one location, it can recognize it in any other location.

In contrast, once a regular DNN has learned to recognize a pattern in one location, it can recognize it only in that particular location.


## TensorFlow Implementation

Convolution layers have quite a few hyperparameters: you must choose the number of filters, their height and width, the strides, and the padding type.

As always, you can use cross-validation to find the right hyperparameter values, but this is very time-consuming.


## Memory Requiremenets

During inference (ie: when making a predition for a new instance) the RAM occupied by one layer can be released as sson as the next layer has been computed, so you only need as much RAM as required by two consecutive layers.

But during training everything computed during the forward pass needs to be preserved for the reverse pass, so the amount of RAM needed is (at least) the total amount of RAM required by all layers.

If training crashes because of an out-of-memory error, you can try reducing the mini-batch size.

Alternatively, you can try reducing dimensionality using a stride, or removing a few layers. 

Or you can try 16-bit floats instead of 32-bit floats. Or you could distribute the CNN across multiple devices.



## Pooling Layer

The goal of pooling is to subsample the input image in order to reduce the computational load, the memory usage, and the number of parameters (thereby limiting the risk of overfitting).

Just like convolution layers, each neuron in a pooling layer is conncted to the outputs of a limited number of neurons in the previous layer, located within a small rectangular receptive field.

However, a pooling neuron has no weights; all it does is aggregate the inouts using an aggregation function such as the max or mean.

Max pooling canoffer a small disount of rotationsla invariance and a slight scale invariance. Such invariance (even if its limited) can be useful in cases where the prediction should not depend on these details, such as oin classification tasks.

But max pooling has some downsides: firstly, it is obniously very destructive: even with a tiny 2 * 2 kernel and a stride of 2, the output will be two times smaller in both direction (so its area will be four times smaller), simply dropping 75% of the input values.

And in some application, invariance is not desirable, for example for sematic segmentation: this is teh task of classifying each pixel in an image depending on the object that pixel belongs to: obviously, if the input image is translated by 1 pixel to the right, the output should alse be translated by 1 pixel to the right.

The goal in this case is equivariance, not invariance: a small change to the outputs should lead to a corresponding change in the output.


## TensorFlow Implementations

Implementing a max pooling layer in TensorFlow is quite easy. The following code created a max pooling layer using a 2x2 kernel.

To create an average pooling layer, just use AvgPool2D instead of MaxPool2D. 

Note that max pooling and average pooling can be performed along the depth dimension rather than the spatial dimensions, although this is not common.

This can allow the CNN to learn to be invariant to various features.

For example, it could learn multiple filters, each detecting a different rotation of the same pattern, such as hand written digits, and depth-wise max pooling layer would ensure taht the output is the same regardless of the rotation.

The CNN could similarly learn to be invariant to anything else: thickness, brightness, skew, color and so on.

One last type of pooling if the global average pooling layer. It works very differently: all it does is compute the mean of each feature map. 

This means that it just outputs a single number per feature map and per instance. Although this is of course extremely destructive, it can be useful as the output layer.

To create such a layer, simply use the keras.layers.GlobalAvgPool2D class:


## CNN Architectures

Typically CNN architectures stack a few convolutions layers (each one generally followed by ReLU layer), then a pooling layer, then another few convolutional layers (+ReLu), then another pooling layer, and so on.

The image gets smaller and smaller as it progesses through the netrowrk, but it also typically gets smaller and smaller as it progresses through the network, but it alsotypically gets deeper and deeper (ie., with more feature maps) thanks to the convolution layer.



## LeNet-5

## AlexNet

It was the first to stack convolution layers directly on top of each other, instead of stacking a pooling layer on top of each convolution layer.

To reduce overfitting the authors used two regualrization techniwues: first they applied dropouts with 50% dropuout rate during trainig to the outputs of layers F8 and F9, tehy performed data augmentation by randomly shifting training images by various offsets, flipping them horizontally and changing the lighting conditions.

Data augmentation artificially increases the size of the training set by generating many realistic variants of each training instance.

This reduced overfitting, making this a regualrization technique.

The generated instance s should be realistic as possible:

AlexNet also uses a competitive normalization step immediately after the ReLU step of layers C! and C3, called local response normalization.


## GoogleNet

The improved performance from it came in large part from the fact that the network was much deeper than the previous CNNs.

This was made possible by sub-networks called inception modukes, which allowed GoogleNet to use parameters much more efficiently than previous architectures: 


## VGGNet 

ResNet :- The ker to being able to train such a deep network is to use skip connections (also called shortcut connections): the signal feeding into a layer is also added to the output of a layer located a bit higher up the stack.


Xception:- It replaces the inception modules with a special type of layer called a depthwise separable convolution,

SENet :- It extends existing architectures such as inception networks or ResNets and boosts their performance.




## Classification and Localization

Localizing an object in a picture can be expressed as a regression task to predict a bounding box around the object, a common approach is to predict the horizontal and vertical coordinates of the object's center, as well as its height and width.

This means we have 4 number s to predict. It does not require much change to the model, we just need to add a second dense output layer with 4 unitrs, and it can be trained using the MSE loss:

The MSE often works fairly well as a cost function to train the model, but its not a great metric to evaluate how ell the model can predict bounding boxes.

The most common metric for this is the Intersection over Union (IoU): it is the area of the overlap between the predicted bounding box and the target bounding box, divided by the area of the union.



## Object Detection

A common appproach was to take a CNN that was trained to classify and locate a single object, then slide it across the image.

A post-processing will be needed to get rid of all the unnecessary bounding boxed. A common approach is called non-max suppression.


This simple approach to object detection works pretty well, but it requires running the CNN many times, so it is quite slow.

Fortunately, there is a much faster way to slide a CNN across an image: using a Fully Convolution Network.

## Fully Convolution Networks (FCNs)

To convert a dense layer to a convolution layer, the number of filters in the convolution layer must be equal to the number of units in the dense layer, the filter size must be equal to the size of the input feature maps, and you must use valid padding.
The stride may be set to one or more.


You Only Look Once (YOLO)

YOLO is an extremely fast and accurate object detection architecture. It is so fast that it can run in realtime on a video.

The choice of detection system depends on many factors: speed, accuracy, available pretrained models, training time, complexity e.tc.


## Semantic Segmentation

In semantic segmentation, each pixel is classified according to the class of the object it belongs to.


## Mean Average Precision (mAP)



