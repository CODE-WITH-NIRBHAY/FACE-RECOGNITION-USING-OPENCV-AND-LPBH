# FACE-RECOGNITION-USING-OPENCV-AND-LBPH


🧠 Features
Real-Time Face Detection: Detects faces in the live webcam feed using Haar Cascades.
Face Recognition: Recognizes faces based on pre-trained data and assigns names to detected faces.
Confidence-Based Recognition: Uses a confidence threshold to identify whether the face is recognized or marked as "Unknown".
Seamless Training and Prediction: Automatically trains the system with a dataset and uses the trained model for predictions.
Save and Load Models: Saves and loads both the trained face recognizer and the label-to-name mapping for easy use.


📦 Requirements
Before running the project, make sure you have these dependencies installed:

OpenCV: For computer vision and face detection.
NumPy: To handle arrays and matrices.
Pickle: For saving and loading the label-to-name dictionary.
You can install the required dependencies by running:

bash
Copy
pip install opencv-python numpy


🏃‍♂️ Setup and Usage
Step 1: Organize Your Dataset
Prepare a dataset with images of the faces you want to recognize. Each person should have their own folder containing their face images. The folder structure should look like this:

markdown
Copy
dataset/
    ├── person_1/
    │    ├── image1.jpg
    │    ├── image2.jpg
    │    └── ...
    ├── person_2/
    │    ├── image1.jpg
    │    ├── image2.jpg
    │    └── ...
    └── ...
You can label these folders with the person's name or an ID (the system will use numeric labels).

Step 2: Train the Model
Run the train_face_recognizer function to train the model using your dataset. This function will create a model and save it as trained_model.yml. It also saves a dictionary (label_dict.pkl) mapping labels to names.

python
Copy
train_face_recognizer("path_to_your_dataset_folder")
Step 3: Run Real-Time Face Recognition
Once the model is trained and saved, you can start the webcam and run the face recognition system. It will detect faces in real-time and recognize them based on the trained model.

python
Copy
# Start the face recognition
video_cap = cv2.VideoCapture(0)
The system will display the webcam feed with names and confidence scores for recognized faces. If a face is recognized with high confidence, it will display the person's name. If the confidence is too low, it will display "Unknown".

🔄 How It Works
Training:

The system collects face images stored in folders for each person.
It trains the LBPH (Local Binary Pattern Histogram) face recognizer using these images and assigns each face a unique numeric label.
The model and label dictionary are saved for later use.
Recognition:

The webcam feed is captured in real-time, and faces are detected using Haar Cascade.
Each detected face is processed and matched with the trained model to predict the person's identity.
The system draws a rectangle around the detected face and displays the name and confidence score.
Confidence Threshold:

If the recognition confidence is higher than a set threshold (e.g., 80%), the name of the person is displayed.
If the confidence is too low, the system labels the person as "Unknown".
💡 Key Concepts
Haar Cascade: A fast object detection algorithm used for face detection in real-time.
LBPH (Local Binary Pattern Histogram): A simple and effective method for face recognition, which works by analyzing the texture of the face and comparing it to a database of known faces.
Confidence Threshold: A metric that determines how confident the system is about its predictions. If the confidence is low, the system will not label the face as a known individual.
📸 Example Usage
1. Training the Model:
Run the following function to train the system with your own dataset:

python
Copy
label_dict = train_face_recognizer('path_to_dataset')
This will train the face recognizer and save the model along with the label-to-name dictionary.

2. Face Recognition in Real-Time:
Once the model is trained, you can run the system in real-time by simply using the webcam:

python
Copy
video_cap = cv2.VideoCapture(0)
while True:
    # Capture and process frames here
    ...
This will show the webcam feed with real-time face detection and recognition.

🧑‍🤝‍🧑 Example Output
When the system detects a face, it will display:

A rectangle around the face.
The name of the person detected (if recognized with high confidence).
A confidence score (indicating how sure the model is about its prediction).
For example:

Person Name: John Doe
Confidence: 95.7%
If the system doesn’t recognize the face, it will display:

Name: Unknown
Confidence: 55.3% (low confidence indicating an unknown person)
📂 Project Structure
Here’s an overview of the project directory:

graphql
Copy
face-recognition-project/
│
├── dataset/                   # Folder containing subfolders of images for each person
│   ├── person_1/              # Folder for person 1
│   ├── person_2/              # Folder for person 2
│   └── ...
├── trained_model.yml          # Saved face recognizer model
├── label_dict.pkl             # Saved dictionary mapping label IDs to person names
├── face_recognition.py        # Main script for training and real-time face recognition
├── requirements.txt           # Required dependencies for the project
└── README.md                  # This file
💡 Future Improvements
Accuracy Tuning: Experiment with different face detection and recognition algorithms to improve accuracy.
Live Streaming Support: Add support for live streaming videos (e.g., using IP cameras or RTSP streams).
Enhanced UI: Build a graphical user interface (GUI) for easier use and management of face datasets.
Multiple Face Recognition: Recognize multiple faces in the same frame and support multiple person identification.
📝 License
This project is licensed under the MIT License. Feel free to use and modify it for personal or commercial purposes.

🤝 Contributing
We welcome contributions! If you have ideas for improving this project or fixing issues, feel free to fork the repository and submit a pull request.

🎉 That's it! You've now set up a real-time face recognition system using OpenCV and LBPH. Enjoy experimenting with face recognition technology! 😎

