import cv2
import streamlit as st
import datetime
import numpy as np
import pandas as pd
from EyeTracker.PupilMeasurement.prediction import predict
from EyeTracker.PupilMeasurement.circler import circler
from EyeTracker.GazeTracking.gazeTracking import GazeTracking
from LLM.LLM import analyze_logs  # Import the analyze_logs function from LLM.py
import io

# Loading models
eye_cascade = cv2.CascadeClassifier('./trained_models/haarcascade_eye_tree_eyeglasses.xml')

def measure_dilation(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

    if len(eyes) == 2:
        ratios = []
        for (x, y, w, h) in eyes:
            roi = gray[y:y+h, x:x+w]
            img = cv2.resize(roi, (256, 256))
            seg = predict(img)  # Use the predict function
            seg = (np.squeeze(seg) * 255).astype('uint8')
            _, seg = cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY)

            circle_params = circler(seg)
            if circle_params:
                ratio, radius_i, center_i, radius_p, center_p = circle_params
                ratios.append((radius_p, radius_i))

        if ratios:
            avg_ratio = sum(r[0] / r[1] for r in ratios) / len(ratios)
            return avg_ratio, ratios

    print("No valid eye detection or ratios calculated.")
    return None, []

def export_logs_and_report_to_csv(logs, llm_report):
    # Create a DataFrame for logs
    logs_df = pd.DataFrame(logs)

    # Add the LLM report as a new column to the logs DataFrame
    logs_df['LLM Report'] = llm_report  # Broadcasting the same report across all rows

    # Convert to CSV
    csv_output = logs_df.to_csv(index=False)
    
    return csv_output

def main():
    st.set_page_config(page_title="Gaze Tracker App")
    st.title("Webcam Display Streamlit App")
    st.caption("Powered by OpenCV, Streamlit")

    if eye_cascade.empty():
        st.error("Error Loading the Haar Cascade file")

    # Initialize Gaze
    gaze = GazeTracking()

    # Initialize the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Columns to display video, LLM report, and logs side by side
    col1, col2 = st.columns(2)
    col3 = st.columns(1)[0]

    # Create placeholders for video feed, LLM report, and logs
    with col1:
        st.subheader("Web Camera")
        frame_placeholder = st.image([])

    with col2:
        st.subheader("LLM Evaluation Report")
        llm_report_placeholder = st.empty()

    with col3:
        st.subheader("Movement and Dilation Log")
        log_placeholder = st.empty()

    stop_button_pressed = st.button("Stop")
    export_button_pressed = st.button("Export Logs and LLM Report as CSV")
    logs = []

    previous_movement = ""
    previous_dilation_state = None

    llm_report = ""

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break

        # Refresh the frame for gaze tracking
        gaze.refresh(frame)
        # Get the annotated frame
        frame = gaze.annotated_frame()

        # Check gaze direction and overlay text
        if gaze.is_blinking():
            movement = "Both eyes blinking"
        elif gaze.is_right():
            movement = "Looking right"
        elif gaze.is_left():
            movement = "Looking left"
        elif gaze.is_center():
            movement = "Looking center"
        else:
            movement = "Unknown"

        # Call the pupil dilation measurement function
        avg_ratio, ratios = measure_dilation(frame)

        # Determine dilation state
        if avg_ratio:
            if avg_ratio > 1:
                dilation_state = 1
                dilation_status = "Dilate"
            elif avg_ratio < 1:
                dilation_state = -1
                dilation_status = "Constrict"
            else:
                dilation_state = 0
                dilation_status = "Normal"
        else:
            dilation_state = 0
            dilation_status = "Normal"

        # Log movement and dilation only if they have changed
        if movement != previous_movement or dilation_state != previous_dilation_state:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {"timestamp": timestamp, "eye movement": movement, "eye dilation status": dilation_status}
            logs.append(log_entry)

            # Update log display
            log_placeholder.text("\n".join([f"{log['timestamp']} | Movement: {log['eye movement']} | Dilation: {log['eye dilation status']}" for log in logs[-10:]]))

            # Analyze logs with the LLM and update the report
            logs_df = pd.DataFrame(logs)
            llm_report = analyze_logs(logs_df)  # Call the analyze_logs function
            llm_report_placeholder.text(llm_report)

            # Update previous states
            previous_movement = movement
            previous_dilation_state = dilation_state

        # Annotate frame
        cv2.putText(frame, movement, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Dilation State: {dilation_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Convert BGR to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        # Stop if button pressed
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Export logs and LLM report if button is pressed
    if export_button_pressed:
        combined_csv_output = export_logs_and_report_to_csv(logs, llm_report)

        # Provide file download for logs and report in CSV format
        st.download_button(
            label="Download Logs and LLM Report",
            data=combined_csv_output,
            file_name="gaze_tracking_logs_and_report.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
