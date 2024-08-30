from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np

class EvaluateModel:

    def __init__(self):
        pass

    def evaluate(self, model, test_data):

        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()

        for batch in test_data.as_numpy_iterator(): 
            X, y = batch
            yhat = model.predict(X)
            pred = []
            for ypred in yhat:
                pred.append(np.argmax(ypred))
            ped = np.array(pred)
            pre.update_state(y, pred)
            re.update_state(y, pred)
            acc.update_state(y, pred)
        
        return pre.result(), re.result(), acc.result()