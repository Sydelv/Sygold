# SyGold Web

Website 1 halaman yang menampilkan prediksi harga emas Antam dari model LSTM
kamu (H+1 s/d H+7), live setiap kali halaman dibuka. Logic prediksi di
`app.py` identik dengan `app.py` versi Streamlit — cuma dibungkus jadi
Flask supaya bisa tampil sebagai halaman HTML sesuai tema desainmu, bukan
tampilan default Streamlit.

## Struktur

```
sygold_web/
├── app.py                      # Flask backend + pipeline prediksi
├── templates/index.html        # Halaman (Jinja2)
├── static/css/style.css        # Tema dark-gold
├── model_h1.keras
├── model_h7.keras
├── feat_scaler.pkl
├── y_scaler_h1.pkl
├── y_scaler_h7.pkl
├── shrinkage_w_h1.pkl         # opsional, dari notebook (lihat catatan di bawah)
├── master_dataset_featured.csv
├── requirements.txt
└── Procfile
```

## Coba di lokal

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Buka `http://127.0.0.1:5000`.

> Butuh Python 3.11 (bukan 3.14) karena TensorFlow belum support versi Python terbaru.

## Deploy ke Render (gratis)

1. Push folder ini ke GitHub repo baru.
2. Buka render.com → New → Web Service → connect repo kamu.
3. Isi:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
4. Deploy. URL publik kamu akan berbentuk `https://sygold-xxxx.onrender.com`.

Catatan: free tier Render "tidur" kalau tidak ada trafik ±15 menit, jadi
request pertama setelah idle bisa lambat (cold start, apalagi model
TensorFlow perlu di-load ulang). Kalau ini masalah buat demo sidang, buka
dulu link-nya 1-2 menit sebelum presentasi.

## Catatan soal GPR

Sama seperti versi Streamlit: nilai GPR di-forward-fill dari data bulanan
terakhir di `master_dataset_featured.csv`, karena sumbernya
(matteoiacoviello.com/gpr.htm) tidak punya API dan harus diunduh manual.
Update `master_dataset_featured.csv` di repo tiap kali ada data GPR baru.

## Catatan soal shrinkage H+1

Evaluasi menunjukkan model H+1 murni sedikit kalah dari baseline naive
(harga besok = harga hari ini). File `shrinkage_w_h1.pkl` berisi bobot `w`
optimal (dicari dari validation set di notebook) untuk blend prediksi
model dengan naive: `prediksi_final = w * prediksi_model + (1-w) * harga_terakhir`.

Setelah menjalankan cell shrinkage di notebook (`project_pi_..._TERFIX.ipynb`),
copy file `shrinkage_w_h1.pkl` yang dihasilkan ke folder ini, sejajar dengan
`app.py`. Kalau file ini tidak ada, `app.py` otomatis fallback ke `w = 1.0`
(prediksi model murni, tanpa shrinkage) — jadi aplikasi tetap jalan normal
walau file ini belum di-generate.
