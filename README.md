# WLKATA ARM Project — Visión por Computadora + Robot Colaborativo

Sistema de **pick & place inteligente** que integra visión por computadora (YOLOv8 + Intel RealSense) con un brazo robótico **WLKATA Mirobot**, para detectar, ubicar y manipular objetos en tiempo real según su clase.

El sistema detecta objetos con un modelo YOLOv8 entrenado a medida, calcula su posición 3D combinando la cámara de profundidad RealSense con una transformación de perspectiva píxel→robot, y ejecuta rutinas automáticas de recogida y depósito (pick & place) según la clase detectada.

## Demo / Funcionamiento

1. La cámara RealSense captura el flujo RGB-D de la escena.
2. YOLOv8 (`best.pt`) detecta y clasifica los objetos sobre el frame RGB, con seguimiento (*tracking*) mediante ByteTrack.
3. Para cada objeto detectado, se toma la profundidad (Z) del punto más cercano dentro de su bounding box y se convierte a la escala de alturas del robot.
4. Las coordenadas en píxeles se transforman a coordenadas del robot (X, Y) mediante una homografía calibrada (`cv2.getPerspectiveTransform`).
5. Al presionar **`e`**, se ordenan los objetos por cercanía al centro y se encola una rutina de pick & place que el brazo ejecuta de forma autónoma (agarre por succión, traslado y depósito según la clase).

## Características principales

- **Detección y tracking en tiempo real** con YOLOv8 + ByteTrack.
- **Percepción 3D** con Intel RealSense (alineación depth↔color y estimación de altura del objeto).
- **Calibración cámara→robot** mediante transformación de perspectiva (homografía) y mapeo de profundidad a altura física.
- **Arquitectura multihilo**: captura de frames, inferencia YOLO, escritura de datos a CSV y control del robot corren en hilos independientes para no bloquear el pipeline de visión.
- **Registro de datos**: exportación a CSV de las detecciones (coordenadas, clase, ranking por cercanía al centro) para trazabilidad y análisis posterior.
- **Rutina de pick & place configurable por clase**, con puntos de depósito (`DROP_BY_CLASS`) y límites de seguridad en Z (`Z_SAFE_MIN` / `Z_SAFE_MAX`).
- **Control manual del brazo por teclado/terminal** para pruebas, homing y calibración de la zona de trabajo.

## Estructura del repositorio

```
wlkta-arm/
├── proyect.py       # Programa principal: visión + robot + CSV (multihilo)
├── muve_test.py     # Prototipo de seguimiento por clase con control manual (teclado)
├── test.py          # Utilidad de consola para mover el brazo y hacer homing manualmente
├── camera.py        # Herramienta para calibrar manualmente los 4 puntos de referencia (píxeles)
├── best.pt           # Pesos del modelo YOLOv8 entrenado
└── images/           # Curvas y métricas del entrenamiento (precisión, recall, mAP, matriz de confusión, etc.)
```

## Requisitos

- Python 3.9+
- Brazo robótico WLKATA Mirobot (conexión UART/USB) y librería `wlkatapython`
- Cámara Intel RealSense (D400 series) y `pyrealsense2`
- Dependencias principales:
  ```bash
  pip install ultralytics opencv-python numpy pyserial pyrealsense2 keyboard
  ```

## Uso

1. **Calibrar la zona de trabajo** (una sola vez, o si se mueve la cámara/el robot): ejecutar `camera.py` para obtener los 4 puntos en píxeles y actualizar `pixel_points` / `robot_points` en `proyect.py`.
2. **Ejecutar el sistema principal**:
   ```bash
   python proyect.py
   ```
3. **Controles durante la ejecución**:
   - `s` → iniciar registro de detecciones en CSV
   - `d` → detener el registro
   - `e` → ordenar objetos detectados por cercanía al centro y ejecutar la rutina de pick & place
   - `ESC` → salir y liberar recursos (cámara, robot, ventanas)

Para pruebas puntuales de movimiento u homing del brazo sin visión, usar `test.py`.

## Modelo de detección

El modelo YOLOv8 (`best.pt`) fue entrenado sobre un dataset propio de las piezas a manipular (carro, cilindro, cubo, carta lado A/B). Tras 200 épocas de entrenamiento, alcanza aproximadamente:

- **Precisión:** ~0.99
- **Recall:** ~1.00
- **mAP50:** ~0.995
- **mAP50-95:** ~0.97

Las curvas de entrenamiento (precisión, recall, F1, matriz de confusión) están disponibles en `images/`.

## Notas y posibles mejoras

- Los parámetros de calibración (`pixel_points`, `robot_points`, límites de Z) están fijos para una configuración específica de cámara/robot; deben recalibrarse si cambia la posición física de alguno.
- El script `muve_test.py` es una versión previa/alternativa orientada a control manual por clase (teclado), útil como referencia pero no es el flujo principal.
- Posibles extensiones: manejo de errores de conexión serial más robusto, parametrización de la calibración vía archivo de configuración, y sustitución de las banderas de teclado (`keyboard`, `cv.waitKey`) por una interfaz de control más flexible.
