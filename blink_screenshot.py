import cv2
import time
# Load eye cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Start camera
cap = cv2.VideoCapture(0)
blink = False   # flag to avoid multiple screenshots
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # ✅ BLINK DETECTION (no eyes detected)
    if len(eyes) ==0:
        if not blink:
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print("📸 Screenshot taken!")

            blink = True
    else:
        blink = False

    # Draw rectangles on eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show warning text
    cv2.putText(frame, "DON'T BLINK 😡", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show camera
    cv2.imshow("Blink = Screenshot 😈", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break
# Release
cap.release()
cv2.destroyAllWindows()