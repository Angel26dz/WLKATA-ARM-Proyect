import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import wlkatapython
import serial
import time
import sys
import numpy as np

# ------------------------- CONFIGURACIONES -------------------------
CENTER_PX = (320, 340)
CENTER_MM = (201, -1)
ESCALA_MM_PX = 0.2691
RECTANGLE_POINTS = [[(86, 138), (583, 125), (583, 367), (101, 387)]]
# Puntos de la imagen en píxeles
pixel_points = np.array([
    [86, 138],     # A
    [583, 125],    # B
    [583, 367],    # C
    [101, 387]     # D
], dtype=np.float32)

# Puntos correspondientes en coordenadas del robot
robot_points = np.array([
    [221, 75],     # A
    [228, -71],    # B
    [150, -72],    # C
    [144, 74]      # D
], dtype=np.float32)

# Diccionarios de seguimiento por clase
m, n, r, a, v1, v2 = {}, {}, {}, {}, {}, {}
name_to_dict = {'m': m, 'n': n, 'r': r, 'a': a, 'v1': v1, 'v2': v2}

# Inicialización de robot y conexión serial
robot = None
serial_conn = None
app = None  # Global para acceso desde process_frame

# ------------------------- FUNCIONES BASE -------------------------
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
    elif r.getState() != "Idle":
        r.homing()

    while r.getState() != "Idle":
        time.sleep(0.25)

    print("Brazo en HOME y listo.")
    return r, ser

def pixel_a_robot(x, y):
    M = cv2.getPerspectiveTransform(pixel_points, robot_points)
    
    punto_pixel = np.array([[[x, y]]], dtype=np.float32)  # Nota el triple array para usar con cv2
    punto_robot = cv2.perspectiveTransform(punto_pixel, M)
    return punto_robot[0][0][0], punto_robot[0][0][1]  # Retorna (X_robot, Y_robot)

def update_class_dicts(class_id, track_id, centroid):
    class_name = model.names[class_id]
    if class_name in name_to_dict:
        name_to_dict[class_name][track_id] = centroid

def muve_to_class(clase):
    if clase not in name_to_dict or not name_to_dict[clase]:
        print(f"[INFO] No se detectó ningún objeto de clase '{clase}'.")
        return

    CLASE_TO_POS = {
        "m": (174, 90, 25),
        "a": (200, 90, 25),
        "r": (220, 90, 25),
        "n": (150, 90, 25),
        "v1": (175, -90, 25),
        "v2": (200, -90, 25)
    }

    if clase not in CLASE_TO_POS:
        print(f"[ERROR] No hay coordenadas de destino para la clase '{clase}'.")
        return

    centro = CENTER_PX
    objetos = name_to_dict[clase]
    track_id, (cx, cy) = min(
        objetos.items(),
        key=lambda item: (item[1][0] - centro[0])**2 + (item[1][1] - centro[1])**2
    )

    x_mm, y_mm = pixel_a_robot(cx, cy)
    print(f"Moviendo al objeto '{clase}' (ID {track_id}) → ({x_mm:.1f}, {y_mm:.1f}) mm")

    # Espera inicial
    while robot.getState() != "Idle":
        time.sleep(0.25)

    # Movimiento al objeto
    robot.writecoordinate(1, 0, x_mm, y_mm, 25, 0, 10, 0)
    while robot.getState() != "Idle":
        time.sleep(0.25)

    # Activar bomba
    robot.pump(1)
    while robot.getState() != "Idle":
        time.sleep(0.25)

    # Subir con el objeto
    robot.writecoordinate(1, 0, 186, 3, 200, 0, 8, 0)
    while robot.getState() != "Idle":
        time.sleep(0.25)

    # Obtener destino
    dest_x, dest_y, dest_z = CLASE_TO_POS[clase]

    # Mover a la posición de depósito
    robot.writecoordinate(1, 0, dest_x, dest_y, dest_z, 0, 8, 0)
    while robot.getState() != "Idle":
        time.sleep(0.25)

    # Soltar objeto
    robot.pump(0)
    while robot.getState() != "Idle":
        time.sleep(0.25)

    # Subir nuevamente
    robot.writecoordinate(1, 0, 186, 3, 200, 0, 8, 0)
    while robot.getState() != "Idle":
        time.sleep(0.25)



def threaded_muve():
    global robot
    if robot.getState() != "Idle":
        print("Esperando al brazo...")
        while robot.getState() != "Idle":
            time.sleep(0.5)
    print("Moviendo brazo al centro físico...")
    robot.writecoordinate(1, 0, 186, 3, 200, 0, 8, 0) #------------------------- centro

def threaded_home():
    global robot
    if robot.getState() == "Alarm":
        robot.homing()
    robot.homing()
    while robot.getState() != "Idle":
        time.sleep(0.25)

# ------------------------- INTERFAZ -------------------------
class App:
    def __init__(self, root):
        global app
        app = self

        self.root = root
        self.root.title("Control Visual del Brazo")

        self.manual_win = None  # <- SOLUCIÓN A ERROR DE VENTANA MANUAL

        self.canvas = tk.Canvas(root, width=640, height=480, bg='blue')
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.coord_label = tk.Label(root, text="Coordenadas: --, --", bg='darkred', fg='white', font=('Arial', 12))
        self.coord_label.grid(row=1, column=0, columnspan=2, sticky='we')

        self.obj_label = tk.Label(root, text="Objeto detectado: -- (X: -- mm, Y: -- mm)", bg='darkgreen', fg='white', font=('Arial', 12))
        self.obj_label.grid(row=2, column=0, columnspan=2, sticky='we', pady=(0, 10))

        self.btn_auto = tk.Button(root, text="Automático", command=self.open_automatic_window)
        self.btn_auto.grid(row=3, column=0, pady=10)

        self.button2 = tk.Button(root, text="Manual", command=self.open_manual_window)
        self.button2.grid(row=3, column=1, pady=10)

        self.running = True
        self.cap = cv2.VideoCapture(1) #camara
        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara.")
            sys.exit(1)
        self.update_video()

    def update_obj_label(self, track_id, x, y):
        self.obj_label.config(text=f"Objeto detectado: ID {track_id} (X: {x:.1f} mm, Y: {y:.1f} mm)")

    def open_automatic_window(self):
        automatic_win = tk.Toplevel(self.root)
        automatic_win.title("Ventana Automática")
        automatic_win.geometry("550x200")

        button_labels = ["Ir al Centro", "Homing", "Mover a m", "Mover a n", "Mover a r", "Mover a a", "Mover a v1", "Mover a v2"]
        button_funcs = [
            lambda: threading.Thread(target=threaded_muve, daemon=True).start(),
            lambda: threading.Thread(target=threaded_home, daemon=True).start(),
            lambda: threading.Thread(target=muve_to_class, args=('m',), daemon=True).start(),
            lambda: threading.Thread(target=muve_to_class, args=('n',), daemon=True).start(),
            lambda: threading.Thread(target=muve_to_class, args=('r',), daemon=True).start(),
            lambda: threading.Thread(target=muve_to_class, args=('a',), daemon=True).start(),
            lambda: threading.Thread(target=muve_to_class, args=('v1',), daemon=True).start(),
            lambda: threading.Thread(target=muve_to_class, args=('v2',), daemon=True).start(),
        ]

        for i in range(8):
            btn = tk.Button(automatic_win, text=button_labels[i], width=15, height=2, command=button_funcs[i])
            btn.grid(row=i // 4, column=i % 4, padx=10, pady=10)

    def open_manual_window(self):
        if self.manual_win and tk.Toplevel.winfo_exists(self.manual_win):
            self.manual_win.lift()
            return

        self.manual_win = tk.Toplevel(self.root)
        self.manual_win.title("Control Manual por Coordenadas")
        self.manual_win.geometry("300x300")

        frame = tk.Frame(self.manual_win)
        frame.pack(pady=10)

        labels = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
        self.entries = {}

        for i, label in enumerate(labels):
            tk.Label(frame, text=f"{label}:").grid(row=i, column=0, padx=5, pady=5)
            entry = tk.Entry(frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[label] = entry

        def enviar():
            try:
                x = float(self.entries['X'].get())
                y = float(self.entries['Y'].get())
                z = float(self.entries['Z'].get())
                rx = float(self.entries['RX'].get())
                ry = float(self.entries['RY'].get())
                rz = float(self.entries['RZ'].get())
            except ValueError:
                messagebox.showerror("Error", "Por favor ingrese valores numéricos válidos.")
                return

            threading.Thread(target=lambda: robot.writecoordinate(1, 0, x, y, z, rx, ry, rz), daemon=True).start()
            #for entry in self.entries.values():
            #    entry.delete(0, tk.END)        ----------------------------------------------------------------------------------- entrys

        tk.Button(self.manual_win, text="Enviar Coordenadas", command=enviar).pack(pady=10)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            processed = process_frame(frame)
            frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=image)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
        if self.running:
            self.root.after(10, self.update_video)

    def update_coords_label(self, x, y):
        self.coord_label.config(text=f"Coordenadas: X: {x:.1f} mm, Y: {y:.1f} mm")

    def on_close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        try:
            robot.pump(0)
            serial_conn.close()
        except Exception as e:
            print(f"[WARN] Al cerrar: {e}")
        self.root.destroy()
    def update_coords_from_robot(self):
        try:
            coords = robot.getCoordinate()
            if coords and len(coords) >= 3:
                x, y, z = coords[0], coords[1], coords[2]
                self.update_coords_label(x, y, z)
        except Exception as e:
            print(f"[ERROR] No se pudo obtener coordenadas del robot: {e}")

        if self.running:
            self.root.after(200, self.update_coords_from_robot)

# ------------------------- PROCESAMIENTO CON YOLO -------------------------
model = YOLO('best.pt')

def process_frame(frame):
    results = model.track(
        source=frame,
        show=False,
        persist=True,
        imgsz=640,
        conf=0.8,
        save=False,
        tracker="bytetrack.yaml"
    )

    # Limpia todos los diccionarios antes de actualizar
    for d in name_to_dict.values():
        d.clear()

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
                class_name = model.names[class_id]
                if class_name in name_to_dict:
                    name_to_dict[class_name][track_id] = (cx, cy)
                    if app:
                        x_mm, y_mm = pixel_a_robot(cx, cy)
                        app.update_coords_label(x_mm, y_mm)
                        app.update_obj_label(track_id, x_mm, y_mm)

    # Dibuja el rectángulo irregular sobre el frame
    pts = np.array(RECTANGLE_POINTS, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return frame


# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    robot, serial_conn = init_robot()
    threading.Thread(target=threaded_home, daemon=True).start()
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
