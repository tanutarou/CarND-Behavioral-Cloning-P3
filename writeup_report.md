# **Behavioral Cloning** 

## Abstract
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./imgs/model.png "Model visualization"
[difficult]: ./imgs/difficult.png "difficult curve"
[mse]: ./imgs/mse.png "mse"
[center]: ./imgs/center.png "center image"
[flip]: ./imgs/flip.png "flipped image"
[origin]: ./imgs/center.png "original Image"
[video1]: ./imgs/video1.gif "video1"
[video2]: ./imgs/video2.gif "video2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video1.mp4 is the driving video around the track 1
* video2.mp4 is the driving video around the track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 90-109) 

I constructed a NN model based on PilotNet[1] which is proposed by NVIDIA in 2016.
The model includes RELU layers to introduce nonlinearity, and the data is normalized and cropped in the model using a Keras lambda layer and Cropping2D (code line 91 and 92). 


#### 2. Attempts to reduce overfitting in the model

The model contains batch normalization layers in order to reduce overfitting. I used normalization for input data. Additionally, batch normalization layers normalize each hidden layer.
They prevent overfitting and the trained model tends to not depend on initial weights.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 77-80). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).
And I used batch_size=64. Larger batch size gets better benefit from batch normalization layers. 64 is the largest value which can be used on my environment.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane clockwise driving, recovering from the left and right sides of the road and counter-clockwise driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model based on PilotNet[1]. I thought this model might be appropriate because it is the famous network for the same task in real.
The self-driving car using PilotNet drove 10 miles on the Garden State Parkway[1]. So I think it is a simple network, but it is also a so powerful network.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a low mean squared error on the validation set. This implied that the model was trained well. The model can drive antonomously one lap around the track 1. But it cannot drive around the track 2 at all.

I gave a lot of thought to it and I found the sentences as bellow in project page.
> Keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

I didn't know it. So the model was trained well in BGR colorspace. But the model predicted in RGB colorspace. It was the reason of my model's fail.

I fixed this problem and run the simulator. My car is fail at this curve(below).

![alt text][difficult]

To combat this, I add batch normalization layers for my model and collect data for this curve more.
At the end of the process, the vehicle is able to drive autonomously around the track 1 and 2 without leaving the road.
I show the mean squared error graph in the training process as bellow.

![alt text][mse]


#### 2. Final Model Architecture

The final model architecture (model.py lines 90-109) s shown below.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Convolution 5x5     	| 2x2 stride, filter:24 	|
| RELU					|												|
|BatchNormalization||
| Convolution 5x5     	| 2x2 stride, filter:36 	|
| RELU					|												|
|BatchNormalization||
| Convolution 5x5     	| 2x2 stride, filter:48 	|
| RELU					|												|
|BatchNormalization||
| Convolution 3x3     	| 1x1 stride, filter:64 	|
| RELU					|												|
|BatchNormalization||
| Convolution 3x3     	| 1x1 stride, filter:64 	|
| RELU					|												|
|BatchNormalization||
| flatten| |
| Fully connected		|  outputs 100       									|
|BatchNormalization||
| Fully connected		|  outputs 50       									|
| Fully connected		|  outputs 10       									|
|BatchNormalization||
| Fully connected		|  outputs 1       									|

Here is a visualization of the architecture (It's very easy with using Keras)

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track 1 and two laps on track 2 using center lane driving. Because, track 2 is more difficult than track 1. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to return the center line.
And, I recorded the inverse rotation driving too.

To augment the data sat, I also flipped images and angles thinking that this would be easily generate different natural data. For example, here is an image that has then been flipped:

![alt text][origin]
![alt text][flip]

And, I used the images from three camera(center, right and left) and shifted angles to get data from left sides and right sides easily.

After the collection process, I had 101,100 number of data points(This includes the augmented data). 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 10 Epochs. Because, training more epochs increase validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.


Bellow gif files are the driving video by my trained model.
Track 1:   
![alt text][video1]

Track 2:  
![alt text][video2]

### Reference
[1] End to End Learning for Self-Driving Cars,
Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba, https://arxiv.org/abs/1604.07316

