import serial
import time
import sys
import os
import csv
from datetime import datetime

import wlkatapython
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import threading
import queue
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN REALSENSE + YOLO
# ============================================================

width, height = 640, 480

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

center_px=(width//2 ,height//2)

# Alineación Depth -> Color 
align = rs.align(rs.stream.color)

pipeline.start(config)

model = YOLO("best.pt")


frame_queue = queue.Queue(maxsize=1)      
result_queue = queue.Queue(maxsize=1)     
stop_event = threading.Event()

# ---- CSV (hilo escritor) ----
record_event = threading.Event()            
csv_queue = queue.Queue()           
csv_stop_event = threading.Event()   

RANK_LOG_DIR="extract"
RANK_LOG_FILE= os.path.join(RANK_LOG_DIR,"rank_log.csv")           

# ============================================================
# COORDENADAS / CALIBRACIÓN
# ============================================================

pixel_points = np.array([
    [234, 152],     # A
    [445, 152],    # B
    [445, 338],    # C
    [234, 338]      # D
], dtype=np.float32)


robot_points = np.array([
    [260, 59],     # A
    [260, -39],    # B
    [163, -39],    # C
    [163, 59]      # D

], dtype=np.float32)

z_max_camera = 155
z_min_camera = 279
z_max_robot = 210
z_min_robot =29

class_1,class_2,class_3,class_4,class_5 = {}, {}, {}, {}, {} 
track_by_class = {}

DROP_BY_CLASS={
    0:(150,150,75),  #carro
    1:(200,-150,50), #cilindro
    2:(150,-150,50), #cubo
    3:(40,150,30),   #carta lado A
    4:(40,150,30)	 #carta lado B
}

WAYPOINT = (190, 5, 210)

pick_queue = queue.Queue()
robot_stop_event = threading.Event()
robot_lock = threading.Lock()

Z_LIFT = 25          # mm para subir antes/después
Z_SAFE_MIN = 29      # tu z_min_robot
Z_SAFE_MAX = 210     # tu z_max_robot
Z_PICK_OFFSET = -5   #ajuste de z
# ============================================================
# ROBOT HELPERS 
# ============================================================

def wait_idle(robot):
    while robot.getState() != "Idle":
        time.sleep(0.05)

def suction_on(robot):
    robot.pump(1)
    wait_idle(robot)

def suction_off(robot):
    robot.pump(0)
    wait_idle(robot)

def move_xyz(robot, x, y, z):
    robot.writecoordinate(1, 0, float(x), float(y), float(z), 0, 20, 0)
    wait_idle(robot)
    
def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def safe_move_xyz(robot, x, y, z):
    move_xyz(robot, x, y, clamp(z, Z_SAFE_MIN, Z_SAFE_MAX))
		
# ============================================================
# ROBOT INIT
# ============================================================
def init_robot(port="/dev/ttyUSB0", baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] No se pudo abrir {port}: {e}")
        sys.exit(1)

    r = wlkatapython.Wlkata_UART()
    r.init(ser, -1)

    if r.getState() == "Alarm" or r.getState() != "Idle":
        r.homing()

    while r.getState() != "Idle":
        time.sleep(0.25)

    print("Brazo en HOME y listo.")
    return r, ser

# ============================================================
# TRANSFORMACIONES
# ============================================================
def pixel_a_robot(x, y):

    M = cv.getPerspectiveTransform(pixel_points, robot_points)
    
    punto_pixel = np.array([[[x, y]]], dtype=np.float32)  # Nota el triple array para usar con cv2
    punto_robot = cv.perspectiveTransform(punto_pixel, M)
    
    return punto_robot[0][0][0], punto_robot[0][0][1]  
    

# ============================================================
# THREAD t2: CAPTURA DE FRAMES
# ============================================================

def get_frame():
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()  # <- IMPORTANTE: alineado

        if not depth_frame or not color_frame:
            continue

        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        if not frame_queue.full():
            frame_queue.put((color, depth))

def z_camera_to_robot(z_cam,
                      z_max_camera=155, z_min_camera=279,
                      z_max_robot=210,  z_min_robot=42):
    # clamp
    if z_cam < z_max_camera:
        z_cam = z_max_camera
    elif z_cam > z_min_camera:
        z_cam = z_min_camera

    # normalizar [0,1]
    t = (z_cam - z_max_camera) / (z_min_camera - z_max_camera)

    # map invertido: 155->210, 279->42
    z_robot = z_max_robot + t * (z_min_robot - z_max_robot)
    return int(round(z_robot))

# ============================================================
# THREAD t3: DETECCIÓN YOLO
# ============================================================
def run_detection():
    while not stop_event.is_set():
        try:
            color, depth = frame_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        results = model.track(
        	source=color,
        	show=False,
        	persist=True,
        	imgsz=640,
        	conf=0.8,
        	save=False,
        	tracker="bytetrack.yaml"
    	     )
        
        if not result_queue.full():
            result_queue.put((results, depth, color))



# ============================================================
# HILO PRINCIPAL: PROCESAMIENTO + VISUALIZACIÓN
# ============================================================

def process_result():
    if result_queue.empty():
        return []

    results, depth, color = result_queue.get()
    frame = results[0]
    img_rgb = frame.plot()

    depth_colored = cv.applyColorMap(cv.convertScaleAbs(depth, alpha=0.5), cv.COLORMAP_JET)

    objects = [] 
	
    if frame.boxes is None or len(frame.boxes) == 0:
        combined = np.hstack((img_rgb, depth_colored))
        cv.imshow("RGB + Depth", combined)
        return objects

    for box in frame.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()

        x1 = max(0, min(x1, depth.shape[1] - 1))
        x2 = max(0, min(x2, depth.shape[1]))
        y1 = max(0, min(y1, depth.shape[0] - 1))
        y2 = max(0, min(y2, depth.shape[0]))
        if x2 <= x1 or y2 <= y1:
            continue

        roi_depth = depth[y1:y2, x1:x2]
        valid_mask = roi_depth > 0
        if not np.any(valid_mask):
            continue

        valid_indices = np.argwhere(valid_mask)
        valid_depths = roi_depth[valid_mask]
        min_pos = int(np.argmin(valid_depths))
        min_y_roi, min_x_roi = valid_indices[min_pos]

        x = x1 + int(min_x_roi)
        y = y1 + int(min_y_roi)
        z_cam = int(depth[y, x])

        z_robot = z_camera_to_robot(z_cam)

        cls_id = int(box.cls[0])
		
        track_id = -1
        if getattr(box, "id", None) is not None and len(box.id) > 0:
            track_id = int(box.id[0])

        objects.append((cls_id, track_id, x, y, z_robot))

        if record_event.is_set():
            try:
                csv_queue.put_nowait((cls_id, track_id, x, y, z_robot))
            except queue.Full:
                pass

    combined = np.hstack((img_rgb, depth_colored))
    cv.imshow("RGB + Depth", combined)

    return objects




# ============================================================
# THREAD t4: ESCRITOR CSV (s inicia / d detiene)
# ============================================================

def extract(output_dir="extract"):
    os.makedirs(output_dir, exist_ok=True)

    f = None
    writer = None

    while not csv_stop_event.is_set():
        try:
            item = csv_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Señal para cerrar archivo actual
        if item is None:
            if f:
                f.flush()
                f.close()
                f = None
                writer = None
            continue

        cls_id ,track_id ,x, y, z = item

        # Solo escribir si record_event está activo
        if not record_event.is_set():
            continue

        # Abrir archivo al primer write (por sesión de grabación)
        if f is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir, f"camera_coords_{ts}.csv")
            f = open(path, "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow(["class_id","x-camera", "y-camera", "z-camera"])

        writer.writerow([cls_id,x, y, z])

    # limpieza
    if f:
        f.flush()
        f.close()
        
def write_rank_csv(batch, out_file=RANK_LOG_FILE):
    os.makedirs(RANK_LOG_DIR, exist_ok=True)
    file_exists = os.path.exists(out_file)

    with open(out_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "rank", "class_id", "track_id", "x_px", "y_px", "z_robot", "d2_center"])

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cx0, cy0 = center_px

        for i, (cls_id, track_id, x, y, z) in enumerate(batch, start=1):
            d2 = (x - cx0)**2 + (y - cy0)**2
            w.writerow([ts, i, cls_id, track_id, x, y, z, d2])        
# ============================================================
# SELECCIÓN / ORDEN
# ============================================================

def chose_target(objects,wanted_cls):
	
    candidates = [o for o in objects if o[0] == wanted_cls]
    
    if not candidates:
    	return None
    
    cx0, cy0 = center_px
    
    return min(candidates,key=lambda o: (o[2]-cx0)**2 + (o[3]-cy0)**2)
    
def sort_by_center(objects):
    cx0, cy0 = center_px
    
    return sorted(objects,key=lambda o: (o[2]-cx0)**2 + (o[3]-cy0)**2)
    
# ============================================================
# THREAD t5: ROBOT RUTINE
# ============================================================

def drop_by_class(cls_id):
    return DROP_BY_CLASS.get(int(cls_id))

def pick_and_place_one(robot, obj):
    cls_id, track_id, x_px, y_px, z_pick = obj

    # pixel -> robot XY
    x_r, y_r = pixel_a_robot(x_px, y_px)

    # drop por clase
    drop = drop_by_class(cls_id)
    if drop is None:
        print(f"[ROBOT] Clase {cls_id} sin DROP. Se omite.")
        return
    dx, dy, dz = drop

    # alturas
    z_pick = clamp(z_pick +Z_PICK_OFFSET, Z_SAFE_MIN, Z_SAFE_MAX)
    z_above = clamp(z_pick + Z_LIFT, Z_SAFE_MIN, Z_SAFE_MAX)

    dz = clamp(dz, Z_SAFE_MIN, Z_SAFE_MAX)
    dz_above = clamp(dz + Z_LIFT, Z_SAFE_MIN, Z_SAFE_MAX)

    # --- Secuencia pick & place ---
    safe_move_xyz(robot, *WAYPOINT)

    # ir sobre el objeto
    safe_move_xyz(robot, x_r, y_r, z_above)
    # bajar
    safe_move_xyz(robot, x_r, y_r, z_pick)

    # agarrar
    suction_on(robot)

    # subir
    safe_move_xyz(robot, x_r, y_r, z_above)
    safe_move_xyz(robot, *WAYPOINT)

    # ir a drop
    safe_move_xyz(robot, dx, dy, dz_above)
    safe_move_xyz(robot, dx, dy, dz)

    # soltar
    suction_off(robot)

    # subir y volver
    safe_move_xyz(robot, dx, dy, dz_above)
    safe_move_xyz(robot, *WAYPOINT)

    print(f"[ROBOT] OK cls={cls_id} id={track_id} pick=({x_r:.1f},{y_r:.1f},{z_pick}) drop=({dx},{dy},{dz})")
	
def robot_worker():
    robot, ser = init_robot()
    robot.homing()
    wait_idle(robot)
    safe_move_xyz(robot, *WAYPOINT)
    print("[ROBOT] Worker listo en WAYPOINT")

    while not robot_stop_event.is_set():
        try:
            batch = pick_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if batch is None:
            break

        for obj in batch:
            if robot_stop_event.is_set():
                break
            with robot_lock:
                pick_and_place_one(robot, obj)

# ============================================================
# THREAD t1: HOMING ROBOT (ARRANQUE)
# ============================================================

def main():
    robot, ser = init_robot()
    robot.homing()
    while robot.getState() != "Idle":
        time.sleep(0.25)

# ============================================================
# EJECUCIÓN
#  - s: iniciar CSV detecciones
#  - d: detener CSV detecciones
#  - e: guardar Orden e iniciar rutina
#  - ESC: salir
# ============================================================
if __name__ == "__main__":
    #t1 = threading.Thread(target=main)          # Homing robot
    t2 = threading.Thread(target=get_frame)     # Captura frames
    t3 = threading.Thread(target=run_detection) # Detección
    t4 = threading.Thread(target=extract)       # CSV writer (tu CSV original)
    t5 = threading.Thread(target=robot_worker, daemon=True) # pick and place
	
    #t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    last_objects = []
    last_e_time = 0.0  # debounce para evitar múltiples "e" por tecla sostenida

    while not stop_event.is_set():
        # IMPORTANTÍSIMO: guardar el retorno
        last_objects = process_result()

        key = cv.waitKey(1) & 0xFF

        if key == 27:  # ESC
            stop_event.set()
            break

        elif key == 115:  # 's' -> iniciar grabación (CSV original)
            record_event.set()
        elif key == 100:  # 'd' -> detener grabación (CSV original)
            record_event.clear()
            try:
                csv_queue.put_nowait(None)  # señal de cierre para el hilo extract
            except queue.Full:
                pass

        elif key == 101:  # 'e' -> guardar rank en CSV aparte (NO toca el CSV original)
            now = time.time()
            if now - last_e_time < 0.35:
                continue
            last_e_time = now

            if not last_objects:
                print("[E] No hay objetos detectados.")
                continue

            batch = sort_by_center(last_objects)   # LISTA ordenada cerca->lejos
            write_rank_csv(batch)                  # rank_log.csv
			
            try:
                pick_queue.put_nowait(batch)       # encola la rutina real
            except queue.Full:
    	        pass
				
    csv_stop_event.set()
    robot_stop_event.set()

    try:
    	csv_queue.put_nowait(None)
    except queue.Full:
    	pass

    try:
    	pick_queue.put_nowait(None)
    except queue.Full:
    	pass

    #t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()

    pipeline.stop()
    cv.destroyAllWindows()
