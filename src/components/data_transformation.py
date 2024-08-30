import tensorflow as tf
import numpy
from tensorflow.keras import layers


class TransformData:

    def __init__(self):
        pass

    def normalize(self, data):

        data_scaled = data.map(lambda x,y: (x/255, y))
        return data_scaled


    
    def split_data(self, data, train=0.7, val=0.2):

        data_size = len(data)
        test = 1-(train+val)

        #train-validation-test split size
        train_size = int(data_size*train)
        val_size = int(data_size*val)
        test_size = int(data_size*test)

        train_data = data.take(train_size)
        val_data = data.skip(train_size).take(val_size)
        test_data = data.skip(train_size+val_size).take(test_size)

        return train_data, val_data, test_data

    def augmentation(self, data, rotation=0.2, zoom=0.2, contrast=0.2):

        # Define data augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),  # Flip horizontally and vertically
            layers.RandomRotation(rotation),                    # Rotate by up to 20%
            layers.RandomZoom(zoom),                        # Zoom in by up to 20%
            layers.RandomContrast(contrast),                    # Adjust contrast by up to 20%
        ])

        augmented_data = data.map(lambda x, y: (data_augmentation(x, training=True), y))

        return augmented_data

