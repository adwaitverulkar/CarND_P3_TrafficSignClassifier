# **Traffic Sign Recognition** 

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

[image1]: ./report_images/distribution.png "Visualization"
[image2]: ./report_images/before.png "Before Grayscaling"
[image3]: ./report_images/after.png "After Grayscaling"
[image4]: ./report_images/architecture.png "Covnet Architecture"
[image5]: ./report_images/0000.png "Traffic Sign 1"
[image6]: ./report_images/0001.png "Traffic Sign 2"
[image7]: ./report_images/0002.png "Traffic Sign 3"
[image8]: ./report_images/0003.png "Traffic Sign 4"
[image9]: ./report_images/0004.png "Traffic Sign 5"
[image10]: ./report_images/0005.png "Traffic Sign 6"


Project Link [project code](https://github.com/adwaitverulkar/CarND_P3_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

The dataset summary was found using NumPy.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distribution amongst the classes.

![alt text][image1]

It can be observed that some classes have significantly more number of training examples than others. This can be improved by augmenting the training data by one or more of the following means.

1. Adding white noise to training images.
2. Flipping (vertically and horizontally) and/or adding distortion.
3. Changing color scheme (in case colored images are being used for training)

### Design and Test a Model Architecture

#### 1. Preprocessing

Firstly, all the images were converted to grayscale, as reducing the number of color channels improves the learning rate of the network. An image of a traffic sign in a very good condition should indicate the same thing as another image of the same sign where the traffic sign is faded or the camera has poor color reproduction due to improper lighting conditions. If the network is trained on colored images, it would take longer for the network to learn that color is of no consequence for most images, a fact that is already known to us.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image3]

Finally the data is mean normalized and intensity values are scaled 0 to 1.

#### 2. CovNet Architecture

The final architecture of the network is shown in the figure below. (courtesy: NN-SVG - Alex Lenail)

![alt text][image4]


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale Image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| Convolution 9x9		| 1x1 stride, valid padding, outputs 20x20x32	|
| Convolution 11x11	    | 1x1 stride, valid padding, outputs 10x10x64	|
| Flatten			    | Input 10x10x64. Output 6400x1.   				|
| Fully connected		| Input 6400x1. Output 1600x1.					|
| Fully connected		| Input 1600x1. Output 400x1.  					|
| Fully connected		| Input 400x1. Output 100x1.					|
| Fully connected		| Input 100x1. Output 43x1.						|
| Sigmoid				| Input 43x1 Logits. Output 43x1 Probabilities.	|
 
**Every subsequent layer is fed the ReLU activation of the previous layer which is passed through a dropout to prevent overfitting.**

#### 3. Model Training

The model was training using the following hyperparameter and functions.

| Hyperparameter/Function        		|     Description	        	| 
|:-------------------------------------:|:-----------------------------:| 
| Optimizer        						| Adam Optimizer				| 
| No. of Epochs    						| 10							|
| Batch Size							| 128							|
| Learning Rate	    					| 0.001							|
| Training Dropout						| 0.75  						|
| Cost Function		    				| Cross-entropy Loss			|
| Final Activation Function			    | Sigmoid  						|


#### 4. Training Pipeline and Design Decisions

* The LeNet-5 architecture gives about 88% accuracy on the validation set and is a good starting point. It was observed that this architecture was underfitting the data set as the training and validation accuracy were low.

* To improve this one additional fully connected layer was added. As max-pooling leaves out a significant portion of the data, all the max pooling layers were replaced by dropouts. This takes care of overfitting, but creates a problem as the input image size doesn't change much in two layers of convolutions. 

* Hence, filter size of the 2nd convolutional layer was increased to 9 and another convolutional layer was added. In the third layer, the filter size chosen was 11. The depth was also increased in successive layers, such that the final depth is more than the number of classes. This successive increase in filter size would gradually help capture more complex features in deeper layers. 

* This improved the validation accuracy to about 97% and the training accuracy to 99.5%. The network is slightly overfit as of now, but will improve with more data. The dropout can be decreased or the network can be L2 regularized, in case of overfitting on the augmented data.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 96.8%
* test set accuracy of 94.2%

#### 5. Suggested Improvements

1. Training data augmentation by adding gaussian white noise, distortions, color shifts (color channels ignored in this network), vertical/horizontal flippings.
2. Adding a gaussian filter to smoothen out images before passing to the network.



### Testing Model on New Images

The model was tested on the following images, which are different from those on test set. They were found using Google Images.


![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

#### 2. Prediction of CovNet on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| Keep right   			| Keep right 											| 
| Road work				| Road work 											|
| No entry				| No entry												|
| General caution		| General caution					 					|
| Children crossing		| Children crossing    									|
| Beware of ice/snow	| End of no passing by vehicles over 3.5 metric tons  	|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%.

#### 3. Prediction Confidence

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right   									| 
| 0.99     				| Road work 									|
| 1.00					| No entry										|
| 0.99	      			| General caution				 				|
| 0.99				    | Children crossing    							|
| 0.13				    | Beware of ice/snow   							|

For the last image the network fails to classify the image correctly. This may be due to the low number of training examples for class 30 (beware of ice/snow).