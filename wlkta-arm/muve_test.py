from ultralytics import YOLO
import cv2 as cv
import wlkatapython
import serial
import time
import sys
import keyboard
import threading

# Cargar modelo YOLO
model = YOLO('best.pt')

# Diccionarios de seguimiento por clase
m, n, r, a, v1, v2 = {}, {}, {}, {}, {}, {}
name_to_dict = {'m': m, 'n': n, 'r': r, 'a': a, 'v1': v1, 'v2': v2}

# Centro del frame (píxeles) y brazo (mm)
CENTER_PX = (320, 340)
CENTER_MM = (201, -1)
ESCALA_MM_PX = 0.2691  # mm/px

# Inicialización de robot y conexión serial
robot = None
serial_conn = None

def update_class_dicts(class_id, track_id, centroid):
    class_name = model.names[class_id]
    if class_name in name_to_dict:
        name_to_dict[class_name][track_id] = centroid

# Conversión de coordenadas píxeles → milímetros
def pixel_a_robot(cx, cy):
    dx_px = cx - CENTER_PX[0]
    dy_px = cy - CENTER_PX[1]
    dx_mm = dx_px * ESCALA_MM_PX
    dy_mm = dy_px * ESCALA_MM_PX
    x_mm = CENTER_MM[0] + dx_mm
    y_mm = CENTER_MM[1] - dy_mm
    return x_mm, y_mm

# Mover el brazo al objeto detectado de una clase específica
# Mover el brazo al objeto detectado más cercano al centro
def muve_to_class(clase):
    if clase not in name_to_dict or not name_to_dict[clase]:
        print(f"[INFO] No se detectó ningún objeto de clase '{clase}'.")
        return

    # Obtener objeto más cercano al centro visual
    centro = CENTER_PX
    objetos = name_to_dict[clase]
    track_id, (cx, cy) = min(
        objetos.items(),
        key=lambda item: (item[1][0] - centro[0])**2 + (item[1][1] - centro[1])**2
    )

    x_mm, y_mm = pixel_a_robot(cx, cy)
    print(f"Moviendo al objeto '{clase}' (ID {track_id}) → ({x_mm:.1f}, {y_mm:.1f}) mm")

    while robot.getState() != "Idle":
        time.sleep(0.25)
    robot.writecoordinate(1, 0, x_mm, y_mm, 30, 0, 10, 0)


# Procesar cada frame
def process_frame(frame):
    results = model.track(
        source=frame,
        show=False,
        persist=True,
        imgsz=640,
        conf=0.65,
        save=False,
        tracker="bytetrack.yaml"
    )

    for result in results:
        frame = result.plot()
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
        cv.line(frame, (0, center_y), (width, center_y), (255, 0, 0), 1)

        if result.obb is not None:
            obb_boxes = result.obb
            vertices = obb_boxes.xyxy
            class_ids = obb_boxes.cls
            track_ids = obb_boxes.id

            for i in range(len(vertices)):
                pts = vertices[i].reshape(-1, 2)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                class_id = int(class_ids[i])
                track_id = int(track_ids[i]) if track_ids is not None else -1
                update_class_dicts(class_id, track_id, (cx, cy))
                cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    return frame

# Inicializar robot
def init_robot(port="COM11", baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] No se pudo abrir {port}: {e}")
        sys.exit(1)

    r = wlkatapython.Wlkata_UART()
    r.init(ser, -1)

    if r.getState() == "Alarm":
        r.homing()

    r.homing()
    while r.getState() != "Idle":
        time.sleep(0.25)

    print("Brazo en HOME y listo.")
    return r, ser

# Hilo para mover al centro físico
def threaded_muve():
    global robot
    if robot.getState() != "Idle":
        print("Esperando al brazo...")
        while robot.getState() != "Idle":
            time.sleep(0.5)
    print("Moviendo brazo al centro físico...")
    robot.writecoordinate(1, 0, 201, -1, 30, 0, 10, 0)

# Hilo para volver a HOME
def threaded_home():
    global robot
    if robot.getState() == "Alarm":
        robot.homing()
    robot.homing()
    while robot.getState() != "Idle":
        time.sleep(0.25)

# Función principal
def main():
    global robot, serial_conn

    robot, serial_conn = init_robot()
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        robot.pump(0)
        serial_conn.close()
        return

    print("Iniciando detección de objetos… (ESC para salir)")
    try:
        while True:
            ret, frame = cap.read(0)
            if not ret:
                print("Frame no válido. Saliendo…")
                break
            output_frame = process_frame(frame)
            cv.imshow('Tracking View', output_frame)

            # Control con teclado
            if keyboard.is_pressed("a"):
                threading.Thread(target=threaded_muve, daemon=True).start()
            elif keyboard.is_pressed("s"):
                threading.Thread(target=threaded_home, daemon=True).start()
            elif keyboard.is_pressed("1"):
                threading.Thread(target=muve_to_class, args=('m',), daemon=True).start()
            elif keyboard.is_pressed("2"):
                threading.Thread(target=muve_to_class, args=('n',), daemon=True).start()
            elif keyboard.is_pressed("3"):
                threading.Thread(target=muve_to_class, args=('r',), daemon=True).start()
            elif keyboard.is_pressed("4"):
                threading.Thread(target=muve_to_class, args=('a',), daemon=True).start()
            elif keyboard.is_pressed("5"):
                threading.Thread(target=muve_to_class, args=('v1',), daemon=True).start()
            elif keyboard.is_pressed("6"):
                threading.Thread(target=muve_to_class, args=('v2',), daemon=True).start()

            if cv.waitKey(1) == 27:  # ESC
                break

            time.sleep(0.1)

    finally:
        cap.release()
        cv.destroyAllWindows()
        robot.pump(0)
        serial_conn.close()
        print("\nDiccionarios finales:")
        print('m =', m)
        print('n =', n)
        print('r =', r)
        print('a =', a)
        print('v1 =', v1)
        print('v2 =', v2)
        print("Recursos liberados. Programa finalizado.")

if __name__ == "__main__":
    main()
