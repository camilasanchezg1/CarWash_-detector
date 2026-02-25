import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO
from twilio.rest import Client

# ===========================
# Configuración de página y estilos
# ===========================
st.set_page_config(page_title="Conteo CarWash", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container, .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stProgress>div>div>div>div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# Configuración de Twilio 
# Utilizamos variables de entorno con fallback para evitar errores de secrets
# ===========================
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "AXXXXXXXXXXXXX")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "XXXXXXXXXXXX")
TWILIO_FROM        = os.getenv("TWILIO_FROM", "+19788003568")
TWILIO_TO          = os.getenv("TWILIO_TO", "+573054004214")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ===========================
# Tracker IOU 
# ===========================
class Tracker:
    def __init__(self):
        self.id_count = 0
        self.trackers = []
        self.max_age = 20
        self.min_hits = 1
        self.iou_threshold = 0.2

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        return wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)

    def update(self, dets):
        updated_tracks = []
        for det in dets:
            matched = False
            for trk in self.trackers:
                if self.iou(det[:4], trk['bbox']) > self.iou_threshold:
                    trk['bbox'] = det[:4]
                    trk['hits'] += 1
                    trk['no_losses'] = 0
                    matched = True
                    break
            if not matched:
                self.trackers.append({'bbox': det[:4], 'hits': 1, 'id': self.id_count, 'no_losses': 0})
                self.id_count += 1

        for trk in self.trackers:
            trk['no_losses'] += 1
        self.trackers = [t for t in self.trackers if t['no_losses'] <= self.max_age and t['hits'] >= self.min_hits]

        for trk in self.trackers:
            updated_tracks.append((*trk['bbox'], trk['id']))
        return updated_tracks

# ===========================
# Funciones de detección y procesamiento
# ===========================
model = YOLO('yolov8s.pt')
det_tracker = Tracker()
LINE_Y = 300
LINE_OFFSET = 5
count_vehicles = set()
count_motorbikes = set()

def detectar_y_filtrar(frame):
    results = model(frame)[0]
    dets = []
    for box in results.boxes:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        if cls_id in [2, 3]:  # 2: coche, 3: moto
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            dets.append((x1, y1, x2, y2, cls_id, conf))
    return dets


def procesa_frame(frame):
    global count_vehicles, count_motorbikes
    h, w, _ = frame.shape
    if w > 1080:
        new_w = 1080
        new_h = int(h * (1080 / w))
        frame = cv2.resize(frame, (new_w, new_h))
        h, w = new_h, new_w

    dets = detectar_y_filtrar(frame)
    dets_array = np.array([[x1, y1, x2, y2, conf] for x1, y1, x2, y2, cls_id, conf in dets])
    tracks = det_tracker.update(dets_array)

    for x1, y1, x2, y2, tid in tracks:
        for bx1, by1, bx2, by2, cls_id, conf in dets:
            if abs(bx1-x1)<8 and abs(by1-y1)<8:
                cy = (y1+y2)//2
                if cy < LINE_Y <= cy + LINE_OFFSET:
                    if cls_id == 2:
                        count_vehicles.add(tid)
                    elif cls_id == 3:
                        count_motorbikes.add(tid)
                break

    for x1, y1, x2, y2, cls_id, conf in dets:
        color = (255, 0, 0) if cls_id == 2 else (0, 0, 255)
        label = f"Carro {conf:.1f}" if cls_id == 2 else f"Moto {conf:.1f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 0), 2)
    cv2.putText(frame, f"Vehiculos: {len(count_vehicles)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Motos: {len(count_motorbikes)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# ===========================
# Interfaz Streamlit
# ===========================
st.title("Conteo de Vehículos y Motos - CarWash")

uploaded_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_path = tfile.name
    output_path = os.path.join(os.getcwd(), "output.mp4")

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    stframe = st.empty()
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = procesa_frame(frame)

        if out is None:
            h, w, _ = processed.shape
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        out.write(processed)

        # mostrar en Streamlit
        stframe.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
        frame_idx += 1
        progress.progress(min(frame_idx/total_frames, 1.0))

    cap.release()
    out.release()

    st.success(f"Procesamiento completado. Vehículos: {len(count_vehicles)}, Motos: {len(count_motorbikes)}")
    st.video(output_path)

    # Enviar resumen por Twilio
    resumen = f"A continuación el Resumen del día en el CarWash fue:\nVehículos detectados: {len(count_vehicles)}\nMotos detectadas: {len(count_motorbikes)}"
    try:
        client.messages.create(body=resumen, from_=TWILIO_FROM, to=TWILIO_TO)
        st.info("Resumen enviado por SMS")
    except Exception as e:
        st.error(f"Error al enviar SMS: {e}")
#streamlit run app.py

# Instrucciones de ejecución 
st.markdown(
"""
**Instrucciones:**
1. Instala las dependencias del archivo  requirements
2. Adjunta el video que deseas detectar
3. Recibe el mensaje a tu app de mensajes de texto.
"""
)
