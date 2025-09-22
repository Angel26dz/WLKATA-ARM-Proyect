import serial
import time
import sys
import wlkatapython

# --- Inicializar el robot ---
def init_robot(port="COM3", baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"[ERROR] No se pudo abrir {port}: {e}")
        sys.exit(1)

    r = wlkatapython.Wlkata_UART()
    r.init(ser, -1)

    # Asegurar que salga de alarma
    if r.getState() == "Alarm" or r.getState() != "Idle":
        r.homing()

    while r.getState() != "Idle":
        time.sleep(0.25)

    print("Brazo en HOME y listo.")
    return r, ser

def main():
    robot, ser = init_robot()

    print("\n=== Control del WLKATA desde terminal ===")
    print("1 → Homing")
    print("2 → Mover a coordenadas (X Y Z)")
    print("3 → Mover a centro")
    print("4 → Salir\n")

    while True:
        opcion = input("Selecciona opción: ").strip().lower()

        if opcion == "1":
            print("➡ Haciendo HOMING...")
            robot.homing()
            while robot.getState() != "Idle":
                time.sleep(0.25)
            print("Homing completado.")

        elif opcion == "2":
            try:
                x = float(input("Ingresa X: "))
                y = float(input("Ingresa Y: "))
                z = float(input("Ingresa Z: "))

                # Esperar a que el robot esté libre
                while robot.getState() != "Idle":
                    time.sleep(0.25)

                print(f"➡ Moviendo a ({x}, {y}, {z})...")
                robot.writecoordinate(1, 0, x, y, z, 0, 8, 0)

                while robot.getState() != "Idle":
                    time.sleep(0.25)
                print(" Movimiento completado.")

            except ValueError:
                print(" Coordenadas inválidas.")

        elif opcion == "4":
            print("Saliendo del programa...")
            break
        elif opcion == "3":
            if robot.getState() != "Idle":
                print("Esperando al brazo...")
                while robot.getState() != "Idle":
                    time.sleep(0.5)
            print("Moviendo brazo al centro físico...")
            robot.writecoordinate(1, 0, 201, -1, 30, 0, 10, 0)
        else:
            print("Opción no válida. Intenta con 1, 2 o 3.")

    ser.close()
    print("🔌 Conexión cerrada.")

if __name__ == "__main__":
    main()
