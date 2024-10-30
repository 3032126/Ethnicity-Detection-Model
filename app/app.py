from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained Keras model
model = tf.keras.models.load_model('trained_model_e100_b64.keras')

# Image preprocessing function (modify this according to your model's input shape)
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize the image to the model's expected input shape (assuming 224x224, modify as needed)
    image = image.resize((224, 224)).convert("RGB")  # Change to match your model's input size
    image = np.array(image)  # Convert the image to a numpy array
    image = image.reshape(1, 224, 224, 3) # Add batch dimension
    return image

@app.get("/")
def hello():
    return "hello fastapi"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make a prediction using the loaded model
        prediction = model.predict(processed_image)
        gender_dict = {0:"Male",1:"Female"}
        race_dict = {0:"White",1:"Black",2:"Asian",3:"Indian",4:"Others"}
        pred_gender = gender_dict[round(prediction[0][0][0])] 
        pred_age = round(prediction[1][0][0])
        pred_race = race_dict[round(np.argmax(prediction[2][0]))]
    
        return JSONResponse(content={"pred_gender": pred_gender,"pred_age":pred_age,"pred_race":pred_race})
        # return  JSONResponse(content={"predicted_class": predicted_class})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)