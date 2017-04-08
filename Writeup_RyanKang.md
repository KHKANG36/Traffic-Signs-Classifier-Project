#**Traffic Sign Recognition** 
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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
###Writeup / README
1. Please refer to my ipyNB & HTML File for the overall result

###Data Set Summary & Exploration
####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  
* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.
The code for this step is contained in the third code cell of the IPython notebook.  

I printed out random train image to figure out the shape

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because in this classification the color does not provide 
much information. It only increase the complexity of calculation with 3 channel. Therefore, I converted it to grayscale
using "cvtColor" method. After that, I normalized the image in order to prevent the overfitting. Finally, I shuffled the train
data.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)
The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

I mostly refer to the Lenet model for traning, validation and testing. However, I used 50 EPOCHS to increase the correctness.
On top of that, I used deeper network for the better accuracy. By increasing the number of output layer for each convolutional 
layer, I can acquire higher accuracy. The output is 800 at "Flatten" layer!!! which are almost twice number of that at Lenet Model 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is described well at ipyNB. 
 
####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 
I also mostly used the Lenet model for this, I got 100% accruray for train data, 95.6% accuracy for validation data.
Finally, I got 93.6% accuracy for the test data.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found the German traffic sign data from the website and it's label is each 
12,14,9,32,40. I tried to choose not too easy to classify and not too difficult to classify as well. 
For example, some of the image has many noise information on the background. I didn't want to test my model at the
best environment.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.
The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 
This is relatively lower than I expected. 
In particular, the model never classify exactly "End of all speed limit" and "roundabout mandatory".
I am not sure about it, but two images have different background information and it prevent the correct classification, I believe.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Even in top 5 softmax probabilities, the correct answer was not included for the last two images. 

Lastly, I visualize my neural network with the aid of outputFeatureMap fucntion. By looking into each stage's featuremap, I could figure out how the characteristic of image can be implemented. In particular, I could understand why we have to use deep neural network to increase the correctness. That's because we can extract a variety of characterics of the images as increasing the number of convolutional filter!!!! 
