import numpy as np 
from visualisations import get_specific_image, get_random_image_from_class


def predicted_testset_labels(model, test_gen): 

    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predicted_classes

def real_testset_labels(model, test_gen): 
    
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    
    test_labels = np.zeros(len(predicted_classes))
    for i in range(0,len(test_gen)) : 
        test_labels[32*i : 32*(i+1)] = np.argmax(test_gen[i][1], axis = 1) 
        
    return test_labels


def prediction(model, path) :
    
    img_to_predict = get_specific_image(path , visualize = False)
    prediction = np.argmax(model.predict(img_to_predict), axis = 1)[0]
    return prediction


def get_cnnlayers_positions_in_model(model):
    
    positions = []
    i =0
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' not in layer.name:
            i+=1
            continue
        positions.append([i, layer.name, layer.filters])
        i+=1
    return positions