import cv2
import face_recognition

# Load the pre-trained face detection model (Haar cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

# Function to recognize faces using face_recognition library
def recognize_faces(image):
    # Find face locations using face_recognition library
    face_locations = face_recognition.face_locations(image)

    # Encode faces
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

# Example of using face detection and recognition
def main():
    # Load an image
    image_path = 'path/to/your/image.jpg'
    image = cv2.imread(image_path)

    # Detect faces
    faces = detect_faces(image)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Recognize faces
    face_locations, face_encodings = recognize_faces(image)

    # Display the results
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('Face Detection and Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
