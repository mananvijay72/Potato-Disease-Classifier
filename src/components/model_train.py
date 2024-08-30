import tensorflow as tf
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
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


        model = Sequential([
            layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=(256,256,3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(units= len(os.listdir('data')), activation='softmax'),
        ])

        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        #logging traing information
        logdir='logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        #training model
        model.fit(train_data, epochs=50, validation_data=val_data, callbacks=[tensorboard_callback, early_stopping])

        return model