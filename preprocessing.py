from tensorflow.keras.preprocessing.image import ImageDataGenerator


def process_data(train_path, test_path, img_dims, batch_size):
    """
    Creat Train set (with data augmentation) and test set using ImageDataGenerator
    Parameters
    ----------
    train_path : path of train set
    test_path : path of test set
    batch_size : batch size considered by the iterator of the generator
    Returns
    -------
    train_gen, test_gen (type : DirectoryIterator) 
    """
    
    # Data generation objects
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        zoom_range = 0.3,
        horizontal_flip = True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # This is fed to the network in the specified batch sizes and image dimensions
    train_gen = train_datagen.flow_from_directory(
        directory = train_path, 
        target_size = (img_dims, img_dims), 
        batch_size = batch_size, 
        class_mode = 'categorical', 
        shuffle=True)
    
    test_gen = test_datagen.flow_from_directory(
        directory=test_path, 
        target_size=(img_dims, img_dims), 
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle=False)
    
    return train_gen, test_gen 