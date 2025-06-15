import cv2 as cv
import numpy as np

# Lista para guardar las coordenadas
clicked_points = []

# Función para manejar los clics
def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Punto {len(clicked_points)}: ({x}, {y})")

# Iniciar la captura de la webcam
cap = cv.VideoCapture(1)

cv.namedWindow("Webcam")
cv.setMouseCallback("Webcam", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar los puntos
    for point in clicked_points:
        cv.circle(frame, point, 5, (0, 0, 255), -1)

    # Dibujar el rectángulo si hay 4 puntos
    if len(clicked_points) == 4:
        pts = np.array(clicked_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    cv.imshow("Webcam", frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27:  # Esc para salir
        break

# Liberar recursos
cap.release()
cv.destroyAllWindows()

# Mostrar puntos
print("\nPuntos seleccionados:")
for i, p in enumerate(clicked_points):
    print(f"Punto {i+1}: {p}")
