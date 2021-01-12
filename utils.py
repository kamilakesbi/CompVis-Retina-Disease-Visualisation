import numpy as np 
from visualisations import get_specific_image, get_random_image_from_class


def predicted_testset_labels(model, test_gen): 
    """
    predicted labels on a given test set generator. 
    Parameters
    ----------
    model : Keras model object
    test_gen : test set generator (type : DirectoryIterator) 
    
    Returns
    -------
    predicted_classes : array with predicted labels 
    """
    
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predicted_classes

def real_testset_labels(test_gen): 
    """
    get real labels from the test set 
    Parameters
    ----------
    test_gen : test set generator (type : DirectoryIterator) 
    Returns
    -------
    test_labels : array with real labels
    """
    
    test_labels = np.zeros(test_gen.samples)
    for i in range(0,len(test_gen)) : 
        test_labels[32*i : 32*(i+1)] = np.argmax(test_gen[i][1], axis = 1) # iteration over 
        
    return test_labels


def prediction(model, path) :
    """
    Use model to predict class of image knowing its path
    Parameters
    ----------
    model : Keras model object
    path : image path 
    Returns
    -------
    prediction : str 
    """
    
    img_to_predict = get_specific_image(path , visualize = False)
    prediction = np.argmax(model.predict(img_to_predict), axis = 1)[0]
    return prediction


def get_cnnlayers_positions_in_model(model):
    """
    get positions, names and number of features of each cnn layer in the model
    Parameters
    ----------
    model : Keras model object
    Returns
    -------
    positions : list containing elements of structure : [position, name, number of layers]
    """
    
    positions = []
    i =0
    for layer in model.layers:
        # check for convolutional layers only 
        if 'conv' not in layer.name:
            i+=1
            continue
        positions.append([i, layer.name, layer.filters])
        i+=1
    return positions