from keras.models import model_from_json
import cv2
import numpy as np
import os
import time

# Load the ASL model
json_file = open("QMixASL.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("QMixASL.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshaping for Keras input
    return feature / 255.0

# Initialize video capture
cap = cv2.VideoCapture(0)

# Label mapping for ASL
label = ['A', 'M', 'N', 'S', 'T', 'blank']

# Define result folder
result_folder = 'results/asl_predictions'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def run_asl_detection(frame):
    # Process frame for ASL detection
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (48, 48))
    cropframe = extract_features(cropframe)
    
    # Prediction
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]
    
    # Show prediction on screen
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accuracy = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accuracy}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save the frame with prediction result to the result folder
    result_path = os.path.join(result_folder, f"frame_{int(time.time())}.jpg")
    cv2.imwrite(result_path, frame)
    
    return frame

def my_main():
    # Start training loop
    while True:
        # Capture the frame from the video
        _, frame = cap.read()

        # Run ASL detection
        frame = run_asl_detection(frame)

        # Display the processed frame
        cv2.imshow("output", frame)

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    my_main()
