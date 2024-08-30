from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn
# Load the pre-trained model

model = load_model(r"models/model.keras")

# Define the FastAPI app
app = FastAPI()

# Define the input data model
class UserInput(BaseModel):
    age: int
    gender: str  # 'male' or 'female'
    hypertension: int  # 0 or 1
    heart_disease: int  # 0 or 1
    smoking_history: str  # 'non_smoker', 'current_smoker', 'past_smoker'
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

# Define the endpoint for prediction
@app.post("/predict/")
def predict(input_data: UserInput):
    # Prepare the input data for prediction
    gender = 0 if input_data.gender == 'male' else 1
    smoking_history_mapping = {'non_smoker': 0, 'current_smoker': 1, 'past_smoker': 2}
    smoking_history = smoking_history_mapping[input_data.smoking_history]

    features = np.array([[
        input_data.age,
        gender,
        input_data.hypertension,
        input_data.heart_disease,
        smoking_history,
        input_data.bmi,
        input_data.HbA1c_level,
        input_data.blood_glucose_level
    ]])

    # Make the prediction
    prediction = model.predict(features)
    
    # Map prediction to a meaningful output
    prediction_output = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    
    return {"prediction": prediction_output}

# To run the app, use this command in the terminal:
# uvicorn test:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5050)

