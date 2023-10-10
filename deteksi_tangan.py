# Import pustaka Tkinter untuk GUI, Label, dan Button
import tkinter as tk
from tkinter import Label, Button
# Import pustaka PIL untuk manipulasi gambar
from PIL import Image, ImageTk
# Import pustaka OpenCV untuk pemrosesan gambar
import cv2
# Import pustaka numpy untuk operasi matematika pada array
import numpy as np
import mediapipe as mp
# from cvzone.FaceMeshModule import FaceMeshDetector

# Fungsi untuk memuat model deteksi objek
def load_object_detection_model():
    net = cv2.dnn.readNetFromCaffe('ssd_files/deploy.prototxt', 'ssd_files/res10_300x300_ssd_iter_140000.caffemodel')
    return net

# Fungsi untuk melakukan deteksi objek pada suatu frame
def perform_object_detection(frame, net):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            object_class = int(detections[0, 0, i, 1])
            detected_objects.append((object_class, confidence, detections[0, 0, i, 3:7]))
    
    return detected_objects

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_upraised_fingers(landmarks):
    # Indeks landmark ujung jari
    tip_ids = [4, 8, 12, 16, 20]

    # Hitung jumlah jari yang terangkat
    upraised_fingers = 0
    for tip_id in tip_ids:
        # Koordinat landmark ujung jari dan pangkal jari
        tip = landmarks[tip_id]
        dip = landmarks[tip_id - 1]

        # Hitung jari terangkat jika ujung jari lebih tinggi dari pangkal jari
        if tip.y < dip.y:
            upraised_fingers += 1

    return upraised_fingers

# Fungsi untuk menggambar garis penghubung antara landmark tangan
def draw_hand_connections(frame, hand_landmarks):
    connections = mp.solutions.hands.HAND_CONNECTIONS
    for connection in connections:
        start_point = connection[0]
        end_point = connection[1]
        x1, y1 = int(hand_landmarks[start_point].x * frame.shape[1]), int(hand_landmarks[start_point].y * frame.shape[0])
        x2, y2 = int(hand_landmarks[end_point].x * frame.shape[1]), int(hand_landmarks[end_point].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Fungsi untuk menjalankan deteksi objek dan tangan secara berkelanjutan
def detect_objects_and_hand():
    # Untuk mengakses kamera pada komputer menggunakan OpenCV
    cap = cv2.VideoCapture(0) # menggunakan kamera default (0)
    net = load_object_detection_model()

    # Membuat objek mediapipe untuk deteksi tangan
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Daftar kelas objek yang mungkin terdeteksi
    object_classes = ["Unknown", "Person"]
    # Warna acak untuk setiap kelas objek
    COLORS = np.random.uniform(0, 255, size=(len(object_classes), 3))
    
    waving_threshold = 10  # Threshold pergerakan tangan untuk melambai
    waving_counter = 0  # Menghitung gerakan tangan

    while True:
        # Untuk membaca Frame dari kamera
        ret, frame = cap.read()
        
        # Flip frame secara horizontal
        frame = cv2.flip(frame, 1)

        # Deteksi tangan menggunakan MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        num_upraised_fingers = 0  # Inisialisasi jumlah jari terangkat

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Menggambar landmark tangan pada frame
                for landmark in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Menggambar garis penghubung antara landmark tangan
                draw_hand_connections(frame, hand_landmarks.landmark)
                
                # Menghitung jumlah jari terangkat pada tangan saat ini
                num_upraised_fingers += count_upraised_fingers(hand_landmarks.landmark)

        # Deteksi gerakan melambai tangan
        if num_upraised_fingers >= 10:  # Jika total jari terangkat dari kedua tangan mencapai atau melebihi 10
            waving_counter += 1
            # if waving_counter >= waving_threshold:
            #     print("Program ditutup.")
            #     app.destroy()
            #     # cap.release()
            #     # cv2.destroyAllWindows()
            #     return
        else:
            waving_counter = 0  # Reset hitungan jika jumlah jari tidak mencukupi

        # Deteksi objek
        detected_objects = perform_object_detection(frame, net)

        for obj_class, confidence, bbox in detected_objects:
            if obj_class < len(object_classes):
                label = object_classes[obj_class]
                color = COLORS[obj_class]

                if label != "Person":
                    label = "Unknown"  # Ganti label dengan "Unknown" untuk objek selain "Person"

                h, w = frame.shape[:2]

                startX, startY, endX, endY = bbox * np.array([w, h, w, h])

                box_width = int(endX - startX)
                box_height = int(endY - startY)
                cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), color, 2)
                y = int(startY) - 15 if int(startY) - 15 > 15 else int(startY) + 15
                # Format akurasi sebagai persentase
                confidence_percentage = f"{confidence * 100:.2f}%"
                cv2.putText(frame, f"{label}: {confidence_percentage}", (int(startX), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Menampilkan jumlah jari yang terdeteksi
        cv2.putText(frame, f"Jari : {num_upraised_fingers}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Menampilkan frame dengan objek yang terdeteksi
        cv2.imshow('Object Detection', frame)

        # Keluar dari loop jika tombol 'c' ditekan
        if cv2.waitKey(1) == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()

app = tk.Tk()
app.title("Object Detection")

# tombol untuk memulai deteksi objek dan tangan
start_button = tk.Button(app, text="Start Detection", command=detect_objects_and_hand)
start_button.pack(pady=10)

# tombol untuk menutup aplikasi
# close_button = tk.Button(app, text="Close", command=app.destroy)
# close_button.pack()

# loop utama antarmuka grafis
app.mainloop()
