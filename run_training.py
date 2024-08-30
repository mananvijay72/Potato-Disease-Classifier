import tensorflow as tf
import numpy as np
from src.pipeline.train_pipeline import begin_trainig
import os



if __name__ =="__main__":
    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)

    model = begin_trainig()

    model_name = "model_potato.keras"
    model.save(os.path.join('models', model_name))
