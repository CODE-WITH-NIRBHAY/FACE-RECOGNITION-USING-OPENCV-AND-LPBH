# Import necessary libraries
import cv2
import os
import numpy as np
import pickle  # We use this to save and load the label-to-name dictionary

# Load a pre-trained face detection model (Haar Cascade)
# This model can detect faces in images or video frames.
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Create a face recognizer using LBPH (Local Binary Pattern Histogram), a simple yet effective method
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Set a threshold for confidence level, below which faces will be labeled as "Unknown"
CONFIDENCE_THRESHOLD = 80

# Function to train the face recognizer using images stored in a folder
def train_face_recognizer(image_folder):
    faces = []   # List to store images of faces
    labels = []  # List to store labels (identifiers for the people)
    label_dict = {}  # Dictionary to map label IDs to person names

    # Loop through each person's folder in the dataset
    for label, person_name in enumerate(os.listdir(image_folder)):
        person_folder = os.path.join(image_folder, person_name)  # Full path to the person's folder
        
        # Check if the folder is indeed a directory (person folder)
        if os.path.isdir(person_folder):
            label_dict[label] = person_name  # Save the name of the person for this label (ID)
            
            # Loop through each image in the person's folder
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)  # Full path to the image
                
                # Only process images with valid extensions (png, jpg, jpeg)
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Read the image in grayscale mode (because color is not necessary for face recognition)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
                if img is not None:
                    faces.append(img)  # Add the image to our faces list
                    labels.append(label)  # Add the corresponding label to the labels list

    # Train the recognizer with the collected faces and labels
    if faces and labels:
        recognizer.train(faces, np.array(labels))  # Train the recognizer model
        recognizer.save('trained_model.yml')  # Save the trained model to a file
        with open('label_dict.pkl', 'wb') as f:  # Save the label-to-name mapping
            pickle.dump(label_dict, f)
        print("Model trained and saved successfully.")
    else:
        print("No valid images found in dataset.")
    
    return label_dict  # Return the dictionary of labels to names

# Try to load a pre-trained model if it exists (so we don't need to train every time)
try:
    recognizer.read('trained_model.yml')  # Load the pre-trained face recognition model
    with open('label_dict.pkl', 'rb') as f:  # Load the label-to-name dictionary
        label_dict = pickle.load(f)
    print("Pre-trained model loaded successfully.")
except:
    print("No pre-trained model found. Training a new model.")
    label_dict = train_face_recognizer("GIVE PATH OF YOUR DATASET FOLDER")  # Train using dataset folder

# Start capturing video from the webcam (real-time face recognition)
video_cap = cv2.VideoCapture(0)  # 0 means the default camera

# Set up the window where the video feed will be displayed
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Start the loop to process each frame from the webcam
while True:
    ret, video_data = video_cap.read()  # Capture a frame from the webcam

    # Check if the frame is grabbed correctly
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    video_data = cv2.flip(video_data, 1)  # Flip the frame horizontally to create a mirror effect

    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    # Detect faces in the frame using the Haar Cascade detector
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Adjust for scale variations
        minNeighbors=5,  # Detect faces with a minimum of 5 neighbors
        minSize=(30, 30)  # Minimum size of faces to detect
    )

    # Loop through all the detected faces
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]  # Extract the Region of Interest (face) from the frame

        # Skip empty faces (in case the ROI has no face data)
        if roi.size == 0:
            continue

        # Use the recognizer to predict the label and confidence for the detected face
        label, confidence = recognizer.predict(roi)

        # Show the person's name if the confidence is below the threshold
        if confidence < CONFIDENCE_THRESHOLD:
            person_name = label_dict.get(label, "Unknown")  # Look up the name from the label dictionary
        else:
            person_name = "Unknown"  # If confidence is low, it's an "Unknown" face

        # Draw a rectangle around the face in the frame
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 0, 225), 2)

        # Display the name and confidence score near the face
        cv2.putText(video_data, f'{person_name} ({confidence:.2f})', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)  # Display text above the face

    # Show the updated video feed with detected faces and names
    cv2.imshow("Face Recognition", video_data)

    # If the user presses the ESC key, exit the loop
    if cv2.waitKey(10) == 27:  # 27 is the ASCII code for the ESC key
        break

# Release the webcam and close any open windows
video_cap.release()
cv2.destroyAllWindows()
