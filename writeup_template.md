# Traffic Sign Recognition

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[visualisation]: ./examples/visualisation.png "Visualisation"
[grayscale]: ./examples/grayscale.png "Grayscaling"
[normalised]: ./examples/normalised.png "Normalised"
[new-sign-1]: ./examples/new-sign-1.png "Traffic Sign 1"
[new-sign-2]: ./examples/new-sign-2.png "Traffic Sign 2"
[new-sign-3]: ./examples/new-sign-3.png "Traffic Sign 3"
[new-sign-4]: ./examples/new-sign-4.png "Traffic Sign 4"
[new-sign-5]: ./examples/new-sign-5.png "Traffic Sign 5"
[softmax]: ./examples/softmax.png "Softmax"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/stevedomin/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**.
* The size of the validation set is **4410**.
* The size of test set is **12630**.
* The shape of a traffic sign image is **32x32**.
* The number of unique classes/labels in the data set is **43**.

#### 2. Include an exploratory visualization of the dataset.

When visualising the dataset I chose to look at the distribution of classes across in each sets (training, validation, test).

![alt text][visualisation]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I wanted to start with something simple and colors only increase the complexity of our model.

![alt text][grayscale]

As a last step, I normalized the image data in order to center the distribution . We want our neural network to be insensitive to the scale of some features (pixel intensity in our case).

![alt text][normalised]

I could also have augmented the training dataset with a library such as imgaug but eventually decided against it as I'm already quite late submitting the project.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on LeNet and consists of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 				|
| Fully connected 1		| outputs 120        									|
| RELU				|        									|
| Dropout				| 0.5        									|
| Fully connected 2		| outputs 84        									|
| RELU				|        									|
| Dropout				| 0.5        									|
| Fully connected 3		| outputs 43        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the network using batched stochastic gradient descent with the Adam optimiser.

I tried several combination of learning rates and epochs and settled on 20 epochs and a learning rate of 0.02.

When I tried using more epochs it didn't lead to any gains in accuracy. I believe as may have reached the limit of the network architecture as is and I would need to do more pre-processing (data augmentation, histogram equalisation) and change the architecture itself in order to get better results.

Something I want to try once I do a second pass on that project is adding a layer of dropout after each convolution.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 96.1%
* test set accuracy of 93.3%

I used LeNet with a kernel size of 3 as my base architecture. We had great results with it in previous lessons so I thought it would be good starting point.

It is highly relevant to the traffic sign recognition problems, in fact Yann LeCunn himself used a variation of it in a paper on the subject (Traffic Sign Recognition with Multi-Scale Convolutional Networks).

The model performed very well early on and the final iteration of the model has an accuracy of 96.1% on the validation set. As you will see below it also performs well on new images.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][new-sign-1] ![alt text][new-sign-2] ![alt text][new-sign-3]
![alt text][new-sign-4] ![alt text][new-sign-5]

The "Children crossing", "Go straight or turn" and "Traffic signal" signs may be difficult to classify because of the few examples available in the training dataset.

I believe the "Children crossing" sign will be the most difficult of them all to recognise because a couple of other signs look similarly whereas both the "Traffic signal" and "Go straight or right" have distinctive features.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Children crossing      		| Beware of ice/snow   									|
| Traffic signals     			| Traffic signals 										|
| Yield					| Yield											|
| Go straight or right	      		| Go straight or right					 				|
| No entry			| No entry      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This does not compare favorably to the accuracy on the test dataset but I suspected the "Children crossing" sign was going to be hard to recognise.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][softmax]

