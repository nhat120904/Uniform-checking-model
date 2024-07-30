import streamlit as st
import cv2
import numpy as np
import time
st.title("FPT Uniform Checking Demo")

# Function to check if there is an orange-like color in the image
def is_orange_present(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range of orange color in HSV
    lower_orange = np.array([15, 150, 150])
    upper_orange = np.array([25, 255, 255])
    
    # Create a mask for orange color
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    
    # Check if there are any non-zero values in the mask
    if cv2.countNonZero(mask) > 0:
        return "Nhân viên mặc đúng đồng phục"
    else:
        return "Nhân viên không mặc đúng đồng phục"

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Display the image
    st.image(image, channels="BGR")
    with st.spinner("Đang kiểm tra nhân viên có mặc đồng phục FPT không..."):
        time.sleep(3)
    # Check for orange color
    result = is_orange_present(image)
    st.write("Kết quả: " + result)
