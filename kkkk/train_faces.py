import cv2
import os

# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Take user ID
user_id = input("Enter your ID (number only): ")

cap = cv2.VideoCapture(0)

count = 0
print("Press Q to stop capturing images")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        
        # Crop face
        face_img = gray[y:y+h, x:x+w]

        # Save image
        file_name = f"dataset/User.{user_id}.{count}.jpg"
        cv2.imwrite(file_name, face_img)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Image {count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Data Collection", frame)

    # Stop after 50 images or press Q
    if cv2.waitKey(1) & 0xFF == ord("q") or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Face images collected successfully!")