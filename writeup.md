# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./writeup_images/traffic_sign_example.png "Visualization"
[image2]: ./writeup_images/traffic_sign_example_2.png "Visualization 2"
[image3]: ./writeup_images/training_data_bar.png "Training Data Bar"
[image4]: ./writeup_images/validation_data_bar.png "Validation Data Bar"
[image5]: ./writeup_images/test_data_bar.png "Test Data Bar"
[image6]: ./writeup_images/new_traffic_signs.png "New German Traffic Signs"
[image7]: ./writeup_images/new_traffic_sign_compare.png "New German Traffic Signs Compare"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The following images show the example images for random label in the training set.

![alt text][image1]
![alt text][image2]

Here is an exploratory visualization of the data set. It is a bar chart showing traffic sign samples containted in Training, Validation and Test dataset. 

![alt text][image3]
![alt text][image4]
![alt text][image5]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the amount of features and thus reduces execution time. Also, converting to grayscale worked well for Sermanet and LeCun as described in their traffic sign classification article

As a last step, I normalized the image data using `(pixel - 128)/ 128` to convert each pixel values [0, 255] to float value with range of [-1, 1] 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model architecture is consist on the LeNet architecture. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 5x5x16		|
| Flatten		        | Outputs 400   	 							|
| Dropout				|												|
| Fully Connected       | Outputs 120									|
| RELU					|												|
| Dropout				|												|
| Fully Connected       | Outputs 84									|
| RELU					|												|
| Dropout				|												|
| Fully Connected       | Outputs 43									|
| Softmax				|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer (already implemented in the LeNet lab). The final settings used were:

* Batch size: 128
* Epochs: 100
* Learning rate: 0.0007
* Mu: 0
* Sigma: 0.1
* Dropout keep probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 0.995 or 99.5%
* Validation set accuracy of 0.967 or 96.7% 
* Test set accuracy of 0.941 or 94.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * I used LeNet as my first architecture from the course. I chose LeNet because it provided a starting point and was already well defined in the class.

* What were some problems with the initial architecture?
    * One of the main problem was the the accuracy was < 90% and some of the traffic sign were not correctly classified. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * The main architecure was not alternered significtly I added few dropouts.

* Which parameters were tuned? How were they adjusted and why?
    * I tuned varies parameters such as epoch and learning rate. By tuning epoch I found the accuracy started to get better. I adjusted the learning rate multiple times fron 0.001. At 0.001 the accuracy was not at my desired level. By changing the learning rate I noticed changes in accuracy. I kept changing learning rate until my desired result was achieved.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * I don't fully understand CNN as of now, I will have to spend more time with this project to understand further about design choices and other functions. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are fifteen German traffic signs that I found on the web:

![alt text][image6]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image7]


The model was able to correctly guess 15 of the 15 traffic signs, which gives an accuracy of 93.33%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The answer to this question is above.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


