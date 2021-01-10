# OCT for retina diseases classification :
#### Over 0.99 accuracy + Interpretability using feature maps/grad-CAM

In this project I develop some useful visualizing tools that could be added to a 0.99 accuracy model for classification of OCT pictures of retina among 4 classes (DME, Normal, Brusen, CNV). 

The data set can be found here (Kaggle) : 
https://www.kaggle.com/paultimothymooney/kermany2018

The idea is to add some interpretability to the classifier. I worked on two aspects : 

- Feature map : I tried to extract some interesting feature maps from convolutional layers that could faciliate visual disease identification. 

- Convolutional activation map (using Grad-CAM) : ni order to visualize what parts of an OCT picture the classifier focuses on when making its prediction. This can be useful to interpret the classifier and quickly detect a risk of misclassification. 

I used a pretrained model found on Kaggle : 
https://www.kaggle.com/plubinski/retinal-separableconv2d-99-validation


