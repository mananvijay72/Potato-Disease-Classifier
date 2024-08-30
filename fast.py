import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

# Load your model
model = tf.keras.models.load_model(r'models\model_potato.keras')

# Preprocess the image
def preprocess_image(image):
    image = image.convert('RGB')
    image = tf.image.resize(image, (256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Prediction function
def prediction(image_array):
    predictions = model.predict(image_array)
    index = np.argmax(predictions.reshape(-1))
    confidence = np.max(predictions)
    class_labels = ["Early Blight", "Late Blight", "Healthy"]
    return class_labels[index], confidence

# Initialize FastAPI app
app = FastAPI()

# Serve HTML form for image upload with styling
@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_content = """
    <html>
        <head>
            <title>Image Prediction</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                    text-align: center;
                    padding: 50px;
                }
                h1 {
                    color: #0066cc;
                }
                form {
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px 0px #aaa;
                    display: inline-block;
                }
                input[type="file"] {
                    margin-bottom: 10px;
                }
                input[type="submit"] {
                    background-color: #0066cc;
                    color: #fff;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #004c99;
                }
            </style>
        </head>
        <body>
            <h1>Upload an Image</h1>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Prediction endpoint with result display and image preview
@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):   
    # Convert image to a NumPy array
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image_array = preprocess_image(image)
    prediction_result, confidence = prediction(image_array)
    
    # Convert image to base64 for display
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Return the HTML response with prediction results and styling
    result_html = f"""
    <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                    text-align: center;
                    padding: 50px;
                }}
                h1 {{
                    color: #0066cc;
                }}
                .result {{
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px 0px #aaa;
                    display: inline-block;
                    margin-top: 20px;
                }}
                img {{
                    margin-top: 20px;
                    max-width: 100%;
                    height: auto;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px 0px #aaa;
                }}
                p {{
                    font-size: 18px;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background-color: #0066cc;
                    color: #fff;
                    text-decoration: none;
                    border-radius: 5px;
                }}
                a:hover {{
                    background-color: #004c99;
                }}
            </style>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <div class="result">
                <p><strong>Prediction:</strong> {prediction_result}</p>
                <p><strong>Confidence:</strong> {round(confidence * 100, 2)}%</p>
                <img src="data:image/png;base64,{img_str}" alt="Uploaded Image">
            </div>
            <a href="/">Upload another image</a>
        </body>
    </html>
    """
    return HTMLResponse(content=result_html)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5050)
