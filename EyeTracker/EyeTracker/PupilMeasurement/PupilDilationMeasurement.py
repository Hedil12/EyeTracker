import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import resize

# Load the U-Net model
model = load_model('with_augm_1-04_model.h5')  # Adjust the model path as needed
target_size = (256, 256)

def predict_eye_region(eye_region):
    # Resize and normalize the eye region for the model
    eye_region_resized = resize(eye_region, target_size)
    eye_region_resized = np.reshape(eye_region_resized, (1, *eye_region_resized.shape, 1))  # Add batch dimension and channel dimension

    # Make prediction
    prediction = model.predict(eye_region_resized)
    return prediction[0, ..., 0]  # Return the first channel of the prediction

def measure_dilation(eye_region):
    # Ensure the eye_region is in the correct format
    if len(eye_region.shape) != 3 or eye_region.shape[2] != 1:
        return None  # Invalid input

    # Predict the pupil region
    prediction = predict_eye_region(eye_region)

    # Threshold to get binary mask
    _, binary_mask = cv2.threshold(prediction, 0.5, 1, cv2.THRESH_BINARY)

    # Calculate the pupil area as the sum of the binary mask
    pupil_area = np.sum(binary_mask)

    # Dummy values for iris size; you may want to replace these with actual values or calculations
    iris_px = 100  # Example iris size in pixels
    pupil_px = pupil_area  # Pupil area from prediction
    pupil_mm = pupil_px / 10  # Example conversion, adjust as needed
    iris_mm = iris_px / 10  # Example conversion, adjust as needed

    return (True, pupil_px, iris_px, pupil_mm, iris_mm, eye_region)  # Returning a tuple for logging
