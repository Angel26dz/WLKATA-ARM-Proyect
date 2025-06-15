import cv2 as cv
from ultralytics import YOLO
import wlkatapython
import serial
import time
import threading
from PIL import Image, ImageTk
import tkinter as tk
import sys

# === PARÁMETROS Y ESTADOS ===
model = YOLO('best.pt')
CENTER_PX = (320, 340)
CENTER_MM = (201, -1)
ESCALA_MM_PX = 0.2691
m, n, r, a, v1, v2 = {}, {}, {}, {}, {}, {}
name_to_dict = {'m': m, 'n': n, 'r': r, 'a': a, 'v1': v1, 'v2': v2}
robot = None
serial_conn = None
current_coords = (0, 0)

# === FUNCIONES BASE ===
def pixel_a_robot(cx, cy):
    dx_px = cx - CENTER_PX[0]
    dy_px = cy - CENTER_PX[1]
    dx_mm = dx_px * ESCALA_MM_PX
    dy_mm = dy_px * ESCALA_MM_PX
    x_mm = CENTER_MM[0] + dx_mm
    y_mm = CENTER_MM[1] - dy_mm
    return x_mm, y_mm

def update_class_dicts(class_id, track_id, centroid):
    class_name = model.names[class_id]
    if class_name in name_to_dict:
        name_to_dict[class_name][track_id] = centroid

def process_frame(frame):
    global current_coords
    results = model.track(source=frame, show=False, persist=True, imgsz=640,
                          conf=0.65, save=False, tracker="bytetrack.yaml")
    for result in results:
        frame = result.plot()
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
                current_coords = pixel_a_robot(cx, cy)

    return frame

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

# === INTERFAZ GRÁFICA ===
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Visual del Brazo")

        self.canvas = tk.Canvas(root, width=640, height=480, bg='blue')
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.coord_label = tk.Label(root, text="Coordenadas: --, --", bg='darkred', fg='white', font=('Arial', 12))
        self.coord_label.grid(row=1, column=0, columnspan=2, sticky='we')

        self.button1 = tk.Button(root, text="Botón 1", width=20)
        self.button2 = tk.Button(root, text="Botón 2", width=20)
        self.button1.grid(row=2, column=0, pady=10)
        self.button2.grid(row=2, column=1, pady=10)

        self.cap = cv.VideoCapture(0)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = process_frame(frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
            self.canvas.image = imgtk  # Keep a reference
            x, y = current_coords
            self.coord_label.config(text=f"Coordenadas: {x:.1f}, {y:.1f}")

        self.root.after(30, self.update_video)

    def on_close(self):
        self.cap.release()
        self.root.destroy()

# === EJECUCIÓN ===
if __name__ == "__main__":
    robot, serial_conn = init_robot()
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

    robot.pump(0)
    serial_conn.close()
