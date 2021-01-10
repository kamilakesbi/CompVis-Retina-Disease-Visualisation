# Optical Coherence Tomography for retina diseases classification : Over 0.99 accuracy + Interpretability using feature maps/grad-CAM

In this project I develop some useful visualizing tools that could be added to a 0.99 accuracy model for classification of Optical Coherence Tomography pictures of retina among 4 classes (DME, Normal, Brusen, CNV). 

The data set can be found here (Kaggle) : 
https://www.kaggle.com/paultimothymooney/kermany2018

The idea is to add some interpretability to the classifier. I worked on two aspects : 

- Feature map : I tried to extract interesting feature maps from convolutional layers that would faciliate visual disease identification. 

- Convolutional activation map (using Grad-CAM) : to visualize what parts of an OCT picture the classifier focuses on in order to make its prediction. This can be useful to interpret the classifier and quickly detect a risk of misclassification. 

In this project I didn't trained my own model but rather used a pretrained model found on Kaggle : 
https://www.kaggle.com/plubinski/retinal-separableconv2d-99-validation


