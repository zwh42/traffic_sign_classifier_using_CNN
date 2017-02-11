#**Traffic Sign Recognition using CNN** 

##Project Summary

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./writeup_image/images_count.png "Visualization"
[image2]: ./writeup_image/preprocessed_sign.png "Preprocessed"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./writeup_image/web_image_raw.png "Traffic Sign 1"
[image5]: ./writeup_image/web_image_result.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

###

You're reading it! and here is a link to my [project code](https://github.com/zwh42/traffic_sign_classifier_using_CNN)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is 32x32 image
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the traffic image count distrubuted. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the **Pre-process the Data Set (normalization, grayscale, etc.)** cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because color is less important than luminance to distinguish the visual  features.Then I normalized the image data because it can improve the contrast.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image2]



####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the **Split Data into Training, Validation and Testing Sets** cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by use the **train_test_split()** function from **sklearn.model_selection**.

My final training set had 31367 number of images. My validation set and test set had 7842 and 12630 number of images.




####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |    32x32x1 Normalized grayscale image    |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 14x14x6       |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 5x5x16        |
|  Fully Connect  |                Output 120                |
|      RELU       |                                          |
|  Fully Connect  |                 Ouput 84                 |
|      RELU       |                                          |
|  Fully Connect  |      Output 43 (number of classes)       |

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the **Train, Validate and Test the Model** of the ipython notebook. 

Hyperparamer and other settings:

|     name      |   value (or choice)   |
| :-----------: | :-------------------: |
| learning rate |         0.001         |
|   optimizer   |    Adam Optimizer     |
| cost function | softmax cross entropy |
|     epoch     |          30           |
|  batch size   |          200          |

#### 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 17th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.981 
* test set accuracy of 0.904


* LeNet architecture was chosen
  * convolutional layer well suited for this problem because that this problem is related to real world image recongination, in which the convolutional networks exploit spatially local correlation by enforcing a local connectivity pattern, and can build up high level hierarchy to extract features for further classfication. 
  * RELU activation layer is used to reduced likelihood of the gradient to vanish
  * Pooling layer can reduce the number of parameters and computation in the network, and hence to also reduce the possibility of overfitting.
  * dropout layer is not used because the training set size is comparely large, based on the network settings (and its number of paramenters), so the risk of overfitting is small that dropout may not be necessary.   
* LeNet was a mature architecture with proved results. The strucutre is relatively simple and easy to understand, and the speed is fast enough. 
* Only 30 epochs is used and with very few parameter tuning effort the accuracy is over 90%, which I think may not be the best but good enough due to the time constrain. 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

The 2nd image of road work and the 7th image of stop may be difficult because their enviroment is complicated and may misleading the classifier.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction (the title of each image is the predicted traffic sign type)

![alt text][image5]



The model was able to correctly guess 5 of the 8 traffic signs, which gives an accuracy of 62.5%. This compares less desired to the accuracy on the test set of over 90%. For the wrong predications, as we mentioned above, the complex enviroment might casue trouble, which I think is the case for the wrong classfication of (road work -> speed limit (30 km/h)), (stop -> priority road). And for the wrong predication speed limit (60 km/h) -> speed limit (80 km/h), the difference between these two signs are small by nature.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

image  1
probability|prediction
1.0000 | Speed limit (70km/h)
0.0000 | Speed limit (30km/h)
0.0000 | Speed limit (20km/h)
0.0000 | Keep left
0.0000 | Wild animals crossing



image  2
probability|prediction
1.0000 | Stop
0.0000 | No entry
0.0000 | Turn left ahead
0.0000 | Keep right
0.0000 | Go straight or right



image  3
probability|prediction
0.5605 | Speed limit (30km/h)
0.2548 | Dangerous curve to the right
0.1268 | Bicycles crossing
0.0568 | Speed limit (20km/h)
0.0011 | Children crossing



image  4
probability|prediction
0.9943 | Speed limit (80km/h)
0.0037 | Speed limit (50km/h)
0.0020 | Speed limit (30km/h)
0.0000 | Speed limit (60km/h)
0.0000 | End of speed limit (80km/h)



image  5
probability|prediction
1.0000 | Road work
0.0000 | Bicycles crossing
0.0000 | Bumpy road
0.0000 | Dangerous curve to the right
0.0000 | Wild animals crossing



image  6
probability|prediction
1.0000 | Speed limit (30km/h)
0.0000 | Speed limit (50km/h)
0.0000 | Speed limit (20km/h)
0.0000 | End of speed limit (80km/h)
0.0000 | Speed limit (80km/h)



image  7
probability|prediction
1.0000 | Yield
0.0000 | Ahead only
0.0000 | No vehicles
0.0000 | Speed limit (60km/h)
0.0000 | No passing



image  8
probability|prediction
0.9178 | Priority road
0.0820 | End of no passing by vehicles over 3.5 metric tons
0.0002 | Speed limit (100km/h)
0.0000 | Roundabout mandatory
0.0000 | Right-of-way at the next intersection

```

```