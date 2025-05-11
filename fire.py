from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO("best.pt")
cap = cv2.VideoCapture("fire3.mp4")
classNames = ["fire"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or failed.")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = math.ceil((box.conf[0] * 100))
            cls = int(box.cls[0])

            # Debug print
            print(f"Detection - Class: {classNames[cls]}, Confidence: {confidence}%")

            if confidence > 30:  # Lowered threshold for testing
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"Box coordinates: {x1}, {y1}, {x2}, {y2}")

                # Try different color and thicker line
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green, thickness=3

                cvzone.putTextRect(
                    frame,
                    f"{classNames[cls]} {confidence}%",
                    [x1 + 8, y1 + 20],
                    scale=1.5,
                    thickness=2,
                    colorR=(0, 0, 0)  # Black background for text
                )

    cv2.imshow("🔥 Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()