import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
from datetime import datetime
import time
from tkinter import messagebox
import Back.neuronas as neuronas
import Back.preparacion_imagenes as prep

CARPETA_CAPTURAS = "./Proyecto/Front/Almacenaje"
Ancho_Video = 750
Alto_Video = 450
ventana_geometri = "900x720"
os.makedirs(CARPETA_CAPTURAS, exist_ok=True)

cap = None
ultimo_guardado = 0

ESTADO_NO_DETECTADO = "NO DETECTADO"
ESTADO_NO_RECONOCIDO = "NO RECONOCIDO"
ESTADO_RECONOCIDO = "RECONOCIDO"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ultimo_guardado = time.time()

def veri(ruta_imagen, modelo):
    img_test = prep.preparacion_img(ruta_imagen)
    if img_test is None:
        return ["Desconocido"], 0.0
    resultado, confianza = neuronas.consultar(modelo, img_test)
    return resultado, confianza

def cerrar():
    global cap
    if cap is not None:
        cap.release()
    ventana.destroy()

def cambiar_estado(texto):
    lbl_estado.config(text=texto)

def guardar_foto(frame):
    nombre = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    ruta = os.path.join(CARPETA_CAPTURAS, nombre)
    cv2.imwrite(ruta, frame)
    return ruta

def conectar_camara():
    global cap
    url = entry_ip.get().strip()
    if not url:
        messagebox.showerror("Error", "Debe ingresar una URL")
        return
    if not (url.startswith("http://") or url.startswith("https://")):
        messagebox.showerror("Error", "La URL debe iniciar con http:// o https://")
        return

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        messagebox.showerror("Error", "No fue posible conectar la cámara IP.")
        cap.release()
        cap = None
        return
    actualizar_video()

def actualizar_video():
    global ultimo_guardado
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            cambiar_estado("ERROR DE CONEXION")
            ventana.after(1000, actualizar_video)
            return

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.2, minNeighbors=10, minSize=(150, 150))

        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(rostros) > 0:
            x, y, w, h = rostros[0]
            rostro = frame[y:y+h, x:x+w]
            tiempo_actual = time.time()

            if (tiempo_actual - ultimo_guardado) > 3:
                ruta = guardar_foto(rostro)
                ultimo_guardado = tiempo_actual
                resultado, confianza = veri(ruta, modelo)

                if confianza >= 0.7:
                    cambiar_estado(f"RECONOCIDO: {resultado[0]} ({confianza:.2f})")
                else:
                    cambiar_estado("NO RECONOCIDO")
        else:
            cambiar_estado(ESTADO_NO_DETECTADO)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(frame_rgb)
        imagen = imagen.resize((Ancho_Video, Alto_Video))
        foto = ImageTk.PhotoImage(image=imagen)
        lbl_video.configure(image=foto)
        lbl_video.image = foto
        ventana.after(20, actualizar_video)

def crear_interfaz(modelo_entrenado):
    global ventana, lbl_video, lbl_estado, entry_ip, modelo
    modelo = modelo_entrenado
    ventana = tk.Tk()
    ventana.title("Reconocimiento Facial")
    ventana.geometry(ventana_geometri)
    ventana.resizable(False, False)

    frame_superior = tk.Frame(ventana)
    frame_superior.pack(pady=10)
    tk.Label(frame_superior, text="IP Cámara:").pack(side=tk.LEFT, padx=5)
    entry_ip = tk.Entry(frame_superior, width=50)
    entry_ip.pack(side=tk.LEFT)
    entry_ip.insert(0, "http://192.168.1.100:8080/video")
    btn_conectar = tk.Button(frame_superior, text="Conectar", command=conectar_camara)
    btn_conectar.pack(side=tk.LEFT, padx=5)

    frame_video = tk.Frame(ventana, width=Ancho_Video, height=Alto_Video)
    frame_video.pack(pady=10)
    frame_video.pack_propagate(False)
    lbl_video = tk.Label(frame_video)
    lbl_video.pack(fill="both", expand=True)

    lbl_estado = tk.Label(ventana, text=ESTADO_NO_DETECTADO, bg="white", fg="black", font=("Arial", 20, "bold"), width=30, height=2, relief="solid")
    lbl_estado.pack(fill="x", padx=10, pady=15)

    ventana.protocol("WM_DELETE_WINDOW", cerrar)
    ventana.mainloop()