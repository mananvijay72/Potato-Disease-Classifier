from src.components.data_ingestion import LoadData
from src.components.data_transformation import TransformData
from src.components.model_train import ModelTrainer
from src.components.model_evaluation import EvaluateModel
import os
def begin_trainig():

    #cleaing and ingestion data
    path = r"data"
    ingestion = LoadData()
    ingestion.clean_data(data_dir=path)
    data = ingestion.ingest_data(data_dir=path)

    #transforming data
    '''
    Normalizing all the data
    Spliting data into train, validation and test
    Augmenting the train data
    '''

    transformer = TransformData()
    data_scaled = transformer.normalize(data=data)
    train_data, val_data, test_data = transformer.split_data(data=data_scaled, train=0.7, val=0.2)
    train_augmented_data = transformer.augmentation(data=train_data)

    #Training the model
    model_trainer = ModelTrainer()
    model = model_trainer.train(train_data=train_augmented_data, val_data=val_data)

    #model metrics
    metrics = EvaluateModel()
    precision, recall, accuracy = metrics.evaluate(model=model, test_data=test_data)
    print("Accuracy:  ", accuracy)
    print("Pecision:  ", precision)
    print("Recall:    ", recall)
    

    return model

