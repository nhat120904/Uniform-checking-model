import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
from inference_sdk import InferenceHTTPClient

# Initialize InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="A7daqzkJiNyEvGwcSepP"
)

# Streamlit app configuration
st.set_page_config(page_title="Uniform Checking Demo", layout="centered")
st.title("Uniform Checking Demo")

st.write("""
This app allows you to upload an image, and it will use a machine learning model to detect uniforms.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # # Convert image to bytes
    # img_bytes = BytesIO()
    # image.save(img_bytes, format='JPEG')
    # img_bytes = img_bytes.getvalue()

    # Perform inference
    result = CLIENT.infer(image, model_id="uniform-detection-gilyg/3")
    st.write(result)
    x = result['predictions'][0]['x']
    y = result['predictions'][0]['y']
    width = result['predictions'][0]['width']
    height = result['predictions'][0]['height']
    label = result['predictions'][0]['class']
    draw = ImageDraw.Draw(image)
    left = x - width / 2
    top = y - height / 2
    right = x + width / 2
    bottom = y + height / 2
    draw.rectangle([left, top, right, bottom], outline="green", width=10)
    draw.text((right, top - 100), label, fill="green")

    st.image(image, caption="Processed Image with Bounding Box", use_column_width=True)
    # # Display inference result
    # if result.status_code == 200:
    #     st.write("Inference Result:")
    #     st.json(result.json())
    # else:
    #     st.write("Error in inference request")
