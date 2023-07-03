import gradio as gr
import cv2
import numpy as np
from keras.models import load_model



class_mapping = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I'
    
}
# Load the model
model = load_model("model1.h5")

# Define the function to recognize the sign
def recognize_sign(image):
    # Preprocess the grayscale image
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(grayscale_image, (64, 64))
    
    # Convert grayscale image to RGB format
    rgb_image = np.repeat(resized_image[..., np.newaxis], 3, axis=-1)
    
    # Make predictions with the model
    prediction = model.predict(np.expand_dims(rgb_image, axis=0))[0]

    predicted_class_index = np.argmax(prediction)
    
    # Map the class index to the corresponding alphabet
    predicted_alphabet = class_mapping[predicted_class_index]
    
    return predicted_alphabet
    

    
    

# Define the input and output interfaces
iface = gr.Interface(fn=recognize_sign, inputs="image", outputs="text",
title="Real-Time Communication System Powered by Ai for Specially Abled ",
description="Upload an image of a sign here. The model will predict the corresponding alphabet in the output section.",
examples=[
        ["C:\\Users\\bvcha\\Desktop\\project\\ASL_Alphabets.png"],
    ],
    css="""
    body {
        font-family: Arial, sans-serif;
        background-color: #f8f8f8;
    }

    .container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        background-color: #fff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    h1 {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }

    .input-label {
        font-weight: bold;
        margin-bottom: 10px;
    }

    .input-field {
        width: 100%;
        padding: 10px;
        font-size: 16px;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
    }

    .button {
        display: block;
        width: 100%;
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        color: #fff;
        background-color: #007bff;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    .result-label {
        font-weight: bold;
        margin-top: 20px;
    }

    .result-text {
        padding: 10px;
        font-size: 16px;
        background-color: #f5f5f5;
        border: 1px solid #ccc;
        border-radius: 4px;
    }

    /* Team Information */

    .team {
        margin-top: 40px;
        text-align: center;
    }

    .team h2 {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .team ul {
        list-style-type: none;
        padding: 0;
        margin
        }

.team li {
  margin-bottom: 5px;
}""",
layout = "vertical"
)
iface.template = "gradio"
iface.extra_html = """
    
    <div class="team">
        <h2>Our Team</h2>
        <ul>
            <li>Niranjan N Nair</li>
            <li>Shree S Nadgauda</li>
            <li>B.V.Chandrahaas</li>
            <li>Vinayak Sai Nalla</li>
            <!-- Add more team members here -->
        </ul>
    </div>
    """




# Launch the interface
iface.launch()
