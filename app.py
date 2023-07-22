import cv2
import datetime
import face_recognition
import csv
import os
import time

# Load the face images
faces = []
names = []
for i in os.listdir("FaceR/faces"):
    for j in os.listdir(os.path.join('FaceR/faces',i)):
        splitt = j.split(".")
        image = face_recognition.load_image_file(f'FaceR/faces/{i}/{splitt[0]}.{splitt[1]}')
        encoding = face_recognition.face_encodings(image)[0]
        faces.append(encoding)
        names.append(splitt[0])

# Initialize the CSV file and list of recorded attendees
with open('attendance.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Time', 'Image File'])
recorded_names = []

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the timer and face counter
face_counter = 0
prev_name = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations,num_jitters=1)

    # Loop through each face in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(faces, face_encoding,tolerance=0.5)
        name = "Unknown"

        # If a match was found and the attendee has not been recorded yet, update the attendance record
        if True in matches:
            index = matches.index(True)
            name = names[index]
            if name not in recorded_names:
                # Increment the face counter if the name matches the previous frame
                if name == prev_name:
                    face_counter += 1
                else:
                    face_counter = 1

                # If the face has been detected in 5 consecutive frames, update the attendance record
                if face_counter >= 6:
                    # Save the frame to a file
                    filename = f"{name}.jpg"
                    cv2.imwrite(filename, frame)

                    # Add the attendance record to the CSV file after 3 seconds
                    
                    with open('attendance.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename])
                    recorded_names.append(name)
                prev_name = name

        # Draw a box around the face and label it
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Attendance System', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()