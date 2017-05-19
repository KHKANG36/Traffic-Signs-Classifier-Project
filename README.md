# **Built and trained a deep neural network model which can classify road traffic signs, using Tensorflow **

This is a project for Udacity Self-Driving Car Nanodegree program. In this project I built and trained a convolutional neural network(CNN) to classify road traffic signs, using TensorFlow. The implementation of the project is in the file `Traffic_Sign_Classifier_RyanKang.ipynb `. 

## Requirement 

- Python > 3.5.0
- OpenCV Library
- TensorFlow > 0.12 
- GPU (if available)

## Run the Project 

You can download pickled dataset for traffic signs from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). 
Then, Run the `Traffic_Sign_Classifier_RyanKang.ipynb` on the ipython notebook. To test a model, you can download any new German traffic signs from website and classify them.  

## About the Project 

In this project, I implemented the pipeline with below steps: 

1. Augment data through various computer vision techniques to protect overfitting  
![Test image](https://github.com/KHKANG36/Traffic-Signs-Classifier-Project/blob/master/Mytest_Image/Aug_Data_Distr.png)
2. Preprocess the data (Grayscaling and Normalization)  
3. Implement CNN classifier which has 2 convolutional layers and 3 fully connected layer (Below image is the feature map from CNN model)
![Test image](https://github.com/KHKANG36/Traffic-Signs-Classifier-Project/blob/master/Mytest_Image/visualize_cnn.png)
4. Train/validate the model (20 EPOCH) and test the model (accuracy of 95.1%)
5. Test the model with new traffic sign images and analyze the result 

More detailed explanation for the project is written on "Writeup_RyanKang.md " file. 

## Discussion/Issues 

1. Relatively low accuracy for new traffic sign images. How we can minimize the overfitting of neural network model? How we can minimize the influence of the background information of traffic sign image? 
