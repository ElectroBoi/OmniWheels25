
import numpy as np
import cv2
import os

# Inisialisasi face recognizer dan face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Percobaan 4.1/training/training_data.yml')
faceCascade = cv2.CascadeClassifier('Percobaan 4.1/classifier/haarcascade_frontalface_default.xml')

# Font untuk teks yang ditampilkan
font = cv2.FONT_HERSHEY_SIMPLEX

# Daftar nama sesuai ID yang digunakan saat training
id = 0
names = ['None', 'NAMA PRAKTIKAN 1', 'NAMA PRAKTIKAN 2', 'NAMA PRAKTIKAN 3']

# Buka kamera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Gagal membuka kamera.")
    exit()

# Set ukuran frame
cam.set(3, 640)  # Lebar
cam.set(4, 480)  # Tinggi

# Ukuran minimum wajah untuk dideteksi
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Loop utama
while True:
    ret, img = cam.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        # Gambar kotak di wajah yang terdeteksi
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Prediksi wajah
        id_pred, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Cek tingkat kepercayaan (semakin kecil nilainya, semakin cocok)
        if confidence < 100:
            id = names[id_pred]
            confidence_text = " {0}%".format(round(100 - confidence))
        else:
            id = "Unknown"
            confidence_text = " {0}%".format(round(100 - confidence))

        # Tampilkan nama dan confidence di frame
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Tampilkan hasil di window
    cv2.imshow('camera', img)

    # Tekan ESC untuk keluar
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Bersihkan setelah keluar
print("\nKeluar Program...")
cam.release()
cv2.destroyAllWindows()