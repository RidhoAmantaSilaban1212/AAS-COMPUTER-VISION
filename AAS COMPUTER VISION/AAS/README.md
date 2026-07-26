# OCR Plat Nomor Kendaraan menggunakan VLM (LM Studio) + Python

Proyek ini melakukan **Optical Character Recognition (OCR)** pada gambar plat nomor
kendaraan menggunakan **Vision-Language Model (VLM)** yang dijalankan secara lokal lewat
[LM Studio](https://lmstudio.ai/), diintegrasikan dengan Python melalui `lmstudio-python` SDK.

Hasil prediksi dievaluasi menggunakan **Character Error Rate (CER)**:

```
CER = (S + D + I) / N
```

- **S** = jumlah substitusi karakter
- **D** = jumlah karakter yang hilang (deletion)
- **I** = jumlah karakter tambahan (insertion)
- **N** = jumlah karakter pada ground truth

S, D, I dihitung dari *backtrace* algoritma edit distance (Levenshtein), bukan hanya
dari total jarak edit, sehingga breakdown-nya akurat.

## 1. Prasyarat

1. **Install LM Studio** dari https://lmstudio.ai/ dan buka aplikasinya.
2. **Download model VLM (multimodal)** lewat LM Studio, misalnya salah satu dari:
   - `qwen2-vl-2b-instruct` (ringan, direkomendasikan untuk laptop)
   - `llava-v1.6-mistral-7b`
   - `bakllava`

   Lewat CLI `lms` (terpasang bersama LM Studio):
   ```bash
   lms get qwen2-vl-2b-instruct
   ```
   Atau cari & download langsung dari tab **Discover** di aplikasi LM Studio (filter: Vision).

3. Pastikan **LM Studio tetap terbuka berjalan di background** — SDK Python akan
   terhubung otomatis ke instance LM Studio yang aktif di komputer yang sama.

4. **Python 3.9+** sudah terpasang.


## 2. Struktur Folder yang Diharapkan

Script mengasumsikan struktur direktori dataset bergaya YOLO seperti berikut:

```
dataset/
├── images/
│   └── test/
│       ├── main.py          <- script ini dijalankan dari sini
│       ├── test001_1.jpg
│       ├── test001_2.jpg
│       └── ...
│   └── train/
│       ├── train001_1.jpg
│       ├── train001_2.jpg
│   └── val/
│       ├── val001_1.jpg
|
└── labels/
|    └── test/
│       ├── test001_1.jpg
│       ├── test001_2.jpg
│       └── ...
│   └── train/
│       ├── train001_1.jpg
│       ├── train001_2.jpg
│   └── val/
│       ├── val001_1.jpg
```

Path label pada script dibentuk dengan `../../labels/test/<nama_file>.txt`, artinya **script harus dijalankan dari dalam folder `images/test/`** (atau folder gambar yang levelnya dua tingkat di bawah folder induk `dataset/`).

Setiap file label `.txt` mengikuti format YOLO (`class_id x_center y_center width height` per baris), dengan `class_id` merepresentasikan karakter:
- `0-9` → digit `0-9`
- `10-35` → huruf `A-Z`

Karakter akan diurutkan berdasarkan `x_center` untuk membentuk string plat nomor ground truth.

## 3. Menjalankan LM Studio

1. Buka aplikasi **LM Studio**.
2. Muat model `qwen.qwen2.5-vl-3b-instruct` (Qwen2.5-VL 3B Instruct).
3. Jalankan **Local Server** dari tab "Developer" / "Local Server".
4. Pastikan server berjalan di `http://127.0.0.1:1234` (port default yang dipakai script).

## 4. Menjalankan Script

Pindah ke folder yang berisi gambar (`images/test/`), lalu jalankan:

```bash
cd dataset/images/test
python main.py
```

Prompt default yang dikirim ke model:
> "What is the license plate number shown in this image? Respond only with the plate number."

Program akan:
1. Membaca setiap gambar di `dataset/images/`
2. Mengirim gambar + prompt ke model VLM lewat LM Studio (`lms.prepare_image` + `chat.add_user_message(..., images=[...])`)
3. Membersihkan teks keluaran model menjadi kandidat plat nomor
4. Menghitung CER terhadap ground truth
5. Menyimpan semua hasil ke `results/predictions.csv` dengan kolom:
   `image, ground_truth, prediction, CER_score`
6. Menampilkan ringkasan (rata-rata CER, jumlah prediksi sempurna, dsb.)

## 5. Output

- **File CSV**: `hasil_evaluasi_ocr.csv` (dibuat di folder tempat script dijalankan), berisi kolom:
  - `image` — nama file gambar
  - `ground_truth` — plat nomor sebenarnya
  - `prediction` — hasil prediksi model (setelah dibersihkan)
  - `CER_score` — skor Character Error Rate per gambar

- **Ringkasan di terminal**, contoh:
  ```
  ==============================
  Total Gambar      : 197
  Prediksi Benar    : 155
  Prediksi Salah    : 42
  Exact Match Acc   : 78.68%
  Average CER       : 0.0451
  Character Acc     : 95.49%
  ==============================
  ```

## 6. Catatan

- Pastikan port `1234` pada LM Studio aktif sebelum menjalankan script, jika tidak seluruh prediksi akan berisi `ERROR: ...`.
- Variabel `dataset_folder = "."` berarti script memindai gambar di folder tempat ia dijalankan — sesuaikan jika struktur folder Anda berbeda.
- Jika ingin mengganti model, ubah nilai `model="qwen.qwen2.5-vl-3b-instruct"` pada fungsi `predict_license_plate()` sesuai nama model yang dimuat di LM Studio.
