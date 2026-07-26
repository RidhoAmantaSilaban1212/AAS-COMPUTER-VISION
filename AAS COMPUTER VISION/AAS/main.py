import sys
print(sys.executable)
import os
import csv
import re
import base64
from openai import OpenAI

# Koneksi ke Local Server LM Studio (Pastikan port 1234 menyala)
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

def calculate_cer(reference, hypothesis):
    """Menghitung Character Error Rate (CER) = (S+D+I)/N"""
    r = reference.replace(" ", "").upper()
    h = hypothesis.replace(" ", "").upper()
    
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
        
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # Deletion
                d[i][j - 1] + 1,      # Insertion
                d[i - 1][j - 1] + cost # Substitution
            )
    
    S_D_I = d[len(r)][len(h)]
    N = len(r)
    
    if N == 0:
        return 0.0 if S_D_I == 0 else 1.0
    return S_D_I / N

def clean_plate(text):
    text = text.upper()

    text = text.replace(
        "THE LICENSE PLATE NUMBER SHOWN IN THE IMAGE IS",
        ""
    )

    text = text.replace('"', "")
    text = text.replace("'", "")

    text = re.sub(r'[^A-Z0-9]', '', text)

    return text.strip()

def normalize_plate(text):
    text = text.upper()

    text = text.replace("O", "0")
    text = text.replace("I", "1")

    return text

def get_ground_truth_from_label(label_path):
    """
    Membaca file label YOLO dan mengubahnya menjadi plat nomor.
    """

    class_map = {
        i: str(i) for i in range(10)
    }

    for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=10):
        class_map[i] = ch

    chars = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 2:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])

            chars.append(
                (
                    x_center,
                    class_map[class_id]
                )
            )

    chars.sort(key=lambda x: x[0])

    plate = "".join(
        char for _, char in chars
    )

    return plate

def predict_license_plate(image_path):
    """Mengirim gambar ke LM Studio dan mengembalikan teks prediksi"""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    try:
        response = client.chat.completions.create(
         model="qwen.qwen2.5-vl-3b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """
                                    "What is the license plate number shown in this image? Respond only with the plate
number."
                                    """
                                      
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


# --- KONFIGURASI FOLDER ---
# Tanda "." berarti membaca gambar di folder yang sama dengan file main.py ini
dataset_folder = "." 
csv_filename = "hasil_evaluasi_ocr.csv"

total_cer = 0
total_images = 0

correct_predictions = 0
wrong_predictions = 0

normalized_correct = 0

# Proses Utama
print("Memulai proses OCR via LM Studio...\n")

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "ground_truth", "prediction", "CER_score"]) 
    
    if not os.path.exists(dataset_folder):
        print(f"Folder '{dataset_folder}' tidak ditemukan!")
    else:
        for filename in os.listdir(dataset_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):

                image_path = os.path.join(dataset_folder, filename)

                label_path = os.path.join(
                    "..",
                    "..",
                    "labels",
                    "test",
                    os.path.splitext(filename)[0] + ".txt"
                )

                gt = get_ground_truth_from_label(label_path)

                print(f"Memproses gambar: {filename}...")

                pred = predict_license_plate(image_path)

                # Bersihkan hasil dari model
                pred = clean_plate(pred)

                cer_score = calculate_cer(gt, pred)
                total_cer += cer_score
                total_images += 1

                if normalize_plate(gt) == normalize_plate(pred):
                    normalized_correct += 1

                if gt == pred:
                    correct_predictions += 1
                    status = "BENAR"
                else:
                    wrong_predictions += 1
                    status = "SALAH"

                writer.writerow([
                    filename,
                    gt,
                    pred,
                    cer_score
                ])

                status = "BENAR" if gt == pred else "SALAH"

                print(
                    f" -> GT: {gt} | "
                    f"Prediksi: {pred} | "
                    f"CER: {cer_score:.2f} | "
                    f"{status}"
                )

                # <-- TAMBAHKAN DI SINI
if total_images > 0:

    avg_cer = total_cer / total_images

    exact_accuracy = (
        correct_predictions / total_images
    ) * 100

    print("\n==============================")
    print(f"Total Gambar      : {total_images}")
    print(f"Prediksi Benar    : {correct_predictions}")
    print(f"Prediksi Salah    : {wrong_predictions}")
    print(f"Exact Match Acc   : {exact_accuracy:.2f}%")
    print(f"Average CER       : {avg_cer:.4f}")
    print(f"Character Acc     : {(1-avg_cer)*100:.2f}%")
    print("==============================")