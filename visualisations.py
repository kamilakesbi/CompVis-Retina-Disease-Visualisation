from tensorflow.keras.preprocessing import image    
import matplotlib.pyplot as plt
import numpy as np 
import os 
import random 
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import cv2



def get_specific_image(path , visualize = False):
    
    """
    Get an image from its path
    Parameters
    ----------
    path : 
    visualize : boolean, to plot or not the image 
    Returns
    -------
    img : tensor of shape (1, 160, 160, 3)
    """

    img = image.load_img(path,target_size=(160,160)) ## load image
    ## plot image 
    if (visualize == True) :
        plt.figure(figsize=(8,8))
        plt.imshow(img)
        label = path.split('/')[-2]
        plt.title('real label : '+ str(label))
    img = np.expand_dims(img,axis=0) /255 ## insert a new axis  and divied by 255 to normalize 
    return img 

def get_random_image_from_class(class_path, visualize = False) : 
    """
    Get an random image from its class path
    Parameters
    ----------
    path : class path
    visualize : boolean, to plot or not the image 
    Returns
    -------
    img : tensor of shape (1, 160, 160, 3)
    """
    ## choose a random image from its class 
    random_file = random.choice([
        x for x in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, x))
    ])
    
    path = class_path + '/'+ str(random_file) 
    img = image.load_img(path,target_size=(160,160)) ## load image 
    if (visualize == True) : # plot image 
        plt.figure(figsize=(8,8))
        plt.imshow(img)
        label = path.split("\\")[-1].split('/')[-2]
        plt.title('real label : '+ str(label))
    img = np.expand_dims(img,axis=0) /255 ## insert a new axis  and divied by 255 to normalize 
    return img 


def plot_filters(model, i) : 
    
    """
    Plot filters convolutional layer in position i in model 
    Parameters
    ----------
    model : Keras object model
    i : int, position of the cnn layer in the model
    Returns
    -------
    no return
    """
    ### usable only for classic convolutional layers
    assert str(model.layers[i].name)[0:4] == 'conv', 'input should be a classic convolutional layer (not separable)'
    
    layer = model.layers[i] ## layer in position i
    filters = layer.get_weights()[0] # get filters 
 
    f_min,f_max = filters.min(),filters.max()
    filters = (filters-f_min)/(f_max-f_min) ## Normalize filters 

    n_filters,ix = layer.filters, 1 ## number of filters in layer 
    
    ## plot filters 
    fig = plt.figure(figsize=(15,15))
    for k in range(n_filters):
        f = filters[:,:,:,k]
        for j in range(3):
            ax = plt.subplot(n_filters,3,ix, )
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:,:,j])
            ix += 1
            
    plt.legend()
    fig.suptitle('Conv layer n° : '+ str(i))
    plt.show()   
    
    
def get_feature_map(model, position, image, visualize = True): 
    """
    get feature maps for a given cnn layer in model for an image. 
    
    Parameters
    ----------
    model : keras object model
    position : int, position of the cnn layer
    image : (in the same format as output of get_specific_image)
    visualize : boolean, to plot or not the image 
    
    Returns
    -------
    feature_map : tensor of shape (1, 160, 160, number of feature maps)
    """
    
    assert 'conv' in model.layers[position].name, 'not a convolutional layer'
    
    conv_output = Model(inputs=model.inputs,outputs=model.layers[position].output) ## convolutional layer
    
    feature_map = conv_output.predict(image) ## feature maps
    
    ## plot all feature_maps of cnn : 
    if (visualize == True) : 
        plt.figure(figsize=(30,30))        
        n = feature_map.shape[-1]//8 ## 8 feature_aps per row

        for i in range(0, n, 1):
            for j in range(1,9, 1):
                ax=plt.subplot(4*n, 8, 8*i+j)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(feature_map[0,:,:, 8*i+j -1 ])
        plt.show()
    
    return feature_map

def heatmap(model, image) : 
    
    """
    plot heatmap of image = (CAM + image)
    Parameters
    ----------
    model : keras object model
    image : (in the same format as output of get_specific_image)
    Returns
    -------
    no return 
    """
    
    ## Grad-CAM steps : 
    predicted = np.argmax(model.predict(image), axis = 1)[0] ## predicted class of image 
    last_conv = model.layers[18] ## last convolutional layer
    ## gradient of the score for predicted class with respects to feature_maps of last conv layer
    grads = K.gradients(model.output[:,predicted],last_conv.output)[0] 
    
    pooled_grads = K.mean(grads, axis = (0,1,2)) ## average pooling over each feature map
    ## function. input : image / output : average pooling coef of fm + feature_maps
    iterate = K.function([model.input],[pooled_grads,last_conv.output[0]]) 
    pooled_grads_value,conv_layer_output = iterate([image]) ## outputs of function 
    
    for i in range(256):
        conv_layer_output[:,:,i] *= pooled_grads_value[i] ## multiplication of each feature map by average pooling coef
    heatmap = np.mean(conv_layer_output,axis=-1) ## mean of weighted feature maps
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            heatmap[x,y] = np.max(heatmap[x,y],0) ## relu 
    
    ## plot feature map : 
    plt.figure(figsize=(10,10))
    upsample = cv2.resize(heatmap, (160,160))
    plt.imshow(image[0])
    plt.imshow(upsample,alpha=0.6)
    classes = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}
    plt.title('predicted class : '+ str(classes[predicted]))
    plt.show()

