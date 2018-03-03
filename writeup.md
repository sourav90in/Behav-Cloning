# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_comb_fin.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
python drive.py model_comb_fin.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the Nvidia Architecture introduced in "End to End Learning for Self-Driving Cars" with a few modifications. The layers of the Sequential model are as follows:

a) First layer is a Cropping2D layer which crops 50 pixels from the top and 20 from the bottom of each image to narrow down the region of interest i.e. the lane as much as possible.
b) The second layer is a Keras Lambda layer which performs normalziation of the input image.
c) The third layer is a Convolution layer with a 5x5 filter and filter depth of 24 with strides as (2,2). This is followed by Batch Normalization and a RELU activation
d) The fourth layer is a Convolution layer with a 5x5 filter and filter depth of 36 with strides as (2,2). This is followed by Batch Normalization and a RELU activation.
e) The fifth layer is a Convolution layer with a 3x3 filter and filter depth of 64 with strides as (1,1). This is followed by Batch Normalization and a RELU activation.
f) The sixth layer is a Convolution layer with a 3x3 filter and filter depth of 64 with strides as (1,1). This is followed by Batch Normalization and a RELU activation.
g) Next is a Flattening layer.
h) The Flattened layer leads to a Dense layer of 100 neurons which is followed by Batch-Normalization, ReLU Activation and Dropout of 0.5.
i) Next is another Dense layer of 50 neurons which is followed by Batch-Normalization, ReLU Activation and Dropout of 0.5.
j)  Next is another Dense layer of 10 neurons which is followed by Batch-Normalization, ReLU Activation.
k)  Finally there is a Dense layer with 1 neuron which provides the regression estimate of the predicted steering angle.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in layers h) and i) mentioned above.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used a combination of two data-sets and collated their Image Data as well as driving logs into a single CSV prior to training:
a) First Data-Set provided for Track-1 by Udacity.
b) Secod Data-Set was data recorded by driving 1.5 laps on Track-2.

I haven't collected any additional data for recovery driving(recovering from left or right sides of the road and used the left and right images instead with a  steering angle offset to achieve this).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to utilize a prior known model that is known to perform well on Steering angle prediction such as the Nvidia architecture and tweak it to suit the data-set derived from the simulator.

In the initial design I started with the data-set provided by Udacity for Track-1 only.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my  model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting and so I added two Dropout layers in the Dense layers of h) and i) mentioned above.

With the above, I could observe that the Model was able to perform very well on Track-1 but performed horribly on Track-2, so I manually collated some data by driving in the Simulator on Track-2 and collating the Image and driving logs for Track-1 and Track-2.

The next step was to try out the performance of the Model on Track-2. Although there was a slight improvement in performance, the vehicle fell off the track after a short start.

I also came across Batch-Normalization technique as a means of Faster Convergence and this looked intuitively similar to Local Response Normalization that I had utilized in the Traffic-Sign classifier, so I added Batch-Normalization to all the layers of the network and tried out the generated model on Track-1 and Track-2. Track-2 driving performance was significantly improved.

Further I added a shuffling step as well prior to splitting the Training and Validation sets to ensure that Validation doesn't contain more of Track-2 data, while Training contained more of Track-1 data.

Also I increased the number of epochs to 15 as I could see that with this model, the training as well as the validation loss kept decreasing upto 15 epochs.

I also tried out throwing away some samples from the Training Data in a random manner with steering angles within 0 to 0.15(since the Historgram showed a huge distribution within this range) so ensure that the Network is not biased towards driving straight more and rather learns to drive curvy roads as well. But that did not significantly help the Track-2 performance so I have disabled it in the final architecture.

I also tried modifying the speed int he drive.py file, but couldn't find a speed lesser than the default speed of 9 good enough to complete Track-2(with lesser speeds, sometimes the car would get stuck or roll-back on upward slopes) so I didn't alter the default speed in the final submission.

At the end of the above steps, Track-1 performance was consistently good, but unfortunately Track-2 performance was sproadically good. During some of the Autonomated driving attempts, Track-2 would get completed successfully but fail at other points of time(fall over at some turns).

#### 2. Final Model Architecture

The layers of the Sequential model are as follows:

a) First layer is a Cropping2D layer which crops 50 pixels from the top and 20 from the bottom of each image to narrow down the region of interest i.e. the lane as much as possible.
b) The second layer is a Keras Lambda layer which performs normalziation of the input image.
c) The third layer is a Convolution layer with a 5x5 filter and filter depth of 24 with strides as (2,2). This is followed by Batch Normalization and a RELU activation
d) The fourth layer is a Convolution layer with a 5x5 filter and filter depth of 36 with strides as (2,2). This is followed by Batch Normalization and a RELU activation.
e) The fifth layer is a Convolution layer with a 3x3 filter and filter depth of 64 with strides as (1,1). This is followed by Batch Normalization and a RELU activation.
f) The sixth layer is a Convolution layer with a 3x3 filter and filter depth of 64 with strides as (1,1). This is followed by Batch Normalization and a RELU activation.
g) Next is a Flattening layer.
h) The Flattened layer leads to a Dense layer of 100 neurons which is followed by Batch-Normalization, ReLU Activation and Dropout of 0.5.
i) Next is another Dense layer of 50 neurons which is followed by Batch-Normalization, ReLU Activation and Dropout of 0.5.
j)  Next is another Dense layer of 10 neurons which is followed by Batch-Normalization, ReLU Activation.
k)  Finally there is a Dense layer with 1 neuron which provides the regression estimate of the predicted steering angle.

#### 3. Creation of the Training Set & Training Process

To capture good driving behaviour, I used the Data set provided by Udacity for Track-1 and collated data from manual driving of 1.5 laps for Track-2.

To augment the data-set, I used the Left and Right camera images as well for training with a Steering angle delta of 0.25 added/subtracted to the steering angle of the centre image. This helped me avoid collecting Recovery data(recovering from side of the lane to the center)

Also instead of driving counter-clock-wise and collecting more image samples, I have augmented the data-set with a flipped image and its corresponding steering angle would be a sign-reversed version of the center image's steering angle.

To summarize, each sample of Training Data ended up generating 4 samples:
a) Center image and its Steering angle.
b) Left camera image and Sterring angle of a) + 0.25
c) Right Camera image and Steering angle of a) - 0.25
d) Flipped image of the center image and its corresponding negative steering angle of a).

I randomly shuffled the entire data-set prior to splitting the Training and Validation data-sets and also incorporated shuffling per batch for the training batch-sets.

I used an adam optimizer so that manually training the learning rate wasn't necessary and also set the number of epochs to 15 for good training and validation losses.
