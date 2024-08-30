import tensorflow as tf
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os


class ModelTrainer:

    def __init__(self):
        pass

    def train(self, train_data, val_data):

        # Define the EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',   # Metric to monitor
            patience=7,           # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
        )

        # Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)


        model = Sequential()
        
        output_units = len(os.listdir('data')) #numbner of labels in our data

        model.add(Conv2D(16, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())

        model.add(Conv2D(32, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())

        model.add(Conv2D(16, (3,3), 1, activation='relu'))
        model.add(MaxPooling2D())

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dense(units=output_units, activation='softmax'))

        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        #logging traing information
        logdir='logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        #training model
        model.fit(train_data, epochs=50, validation_data=val_data, callbacks=[tensorboard_callback, early_stopping])

        return model