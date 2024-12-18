from ultralytics import YOLO
import cv2

# Загружаем предобученную модель
model = YOLO("runs/detect/train_n/weights/best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Камера(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Детектирование объектов
    results = model(frame)

    # Отображение результатов
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Realtime Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()