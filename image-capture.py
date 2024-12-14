import cv2
import os
import asyncio
from deepface import DeepFace
import concurrent.futures

# Load pre-trained face detector from OpenCV (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory to store face images and embeddings
db_path = 'face_db'
if not os.path.exists(db_path):
    os.makedirs(db_path)

# Create a named window
cv2.namedWindow("Face Capture", cv2.WINDOW_AUTOSIZE)

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a list to store face data during the webcam session
collected_faces = []

# Function to capture frames asynchronously using threads
async def capture_frame(executor):
    loop = asyncio.get_event_loop()
    
    while True:
        # Capture frame in a separate thread to avoid blocking
        ret, frame = await loop.run_in_executor(executor, cap.read)
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Resize frame
        frame_small = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Clear the collected_faces list to avoid redundant captures
        collected_faces.clear()
        
        for idx, (x, y, w, h) in enumerate(faces):
            # Extract face from the resized frame
            face_img = frame_small[y:y+h, x:x+w]
            collected_faces.append((face_img, idx))  # Store face with index
            
            # Draw rectangle around the face and label it
            cv2.rectangle(frame_small, (x, y), (x+w, y+h), (200, 0, 0), 2)
            cv2.putText(frame_small, f'Face {idx+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
        
        # Display the camera feed with rectangles
        cv2.imshow('Face Capture', frame_small)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Asynchronous function to process face data after the webcam is closed
async def process_faces(executor):
    loop = asyncio.get_event_loop()

    for face_img, idx in collected_faces:
        # Ask for user name in a separate thread (non-blocking)
        name = await loop.run_in_executor(executor, input, f"Enter the name for Face {idx+1}: ")
        
        face_path = f"{db_path}/{name}.jpg"
        
        # Save face image in a separate thread
        await loop.run_in_executor(executor, cv2.imwrite, face_path, face_img)
        print(f"Saved {name}'s face.")
        
        # Generate and store the facial embedding using DeepFace (non-blocking)
        await loop.run_in_executor(executor, lambda: DeepFace.represent(img_path=face_path, model_name="VGG-Face", enforce_detection=False))
        print(f"Encoded {name}'s face.")

# Main asynchronous function to handle both tasks
async def main():
    executor = concurrent.futures.ThreadPoolExecutor()
    
    # Capture faces while webcam is running
    await capture_frame(executor)
    
    # Process collected faces after the webcam session ends
    if collected_faces:  # Check if any faces were collected
        await process_faces(executor)

if __name__ == "__main__":
    asyncio.run(main())
