import cv2
from AI_Camera.ai_engine import create_engine

engine = create_engine(use_face_recognition=False, use_lbph=False)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame, faces = engine.detect_and_recognize(frame)

    cv2.imshow("AI Engine Test", annotated_frame)
    print("Detected faces:", faces)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
