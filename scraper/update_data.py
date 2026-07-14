"""
Auto-update master_dataset_featured.csv.

Dijalankan otomatis tiap hari oleh GitHub Actions (lihat
.github/workflows/update-data.yml), tapi juga bisa dijalankan manual:

    python scraper/update_data.py

Scraping harga emas + USD/IDR untuk hari-hari yang belum ada di CSV,
recompute semua fitur turunan, lalu simpan balik ke
master_dataset_featured.csv. GPR (GPRD) di-forward-fill dari nilai
terakhir karena sumbernya (matteoiacoviello.com) tidak auto-update dan
harus di-download manual tiap bulan.
"""
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpr_source import fetch_gpr_daily  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / 'master_dataset_featured.csv'

FEATURES_BASE_COLS = ['Date', 'Gold Price', 'USDIDR', 'GPRD', 'GPRD_MA7', 'GPRD_MA30']

BULAN_INDO = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}


def log(msg):
    print(f"[update_data] {msg}", flush=True)


def scrape_gold_price(date: datetime):
    """Sama seperti versi di app.py — cari baris Satuan=1 di kolom Antam,
    bukan posisi index tetap, biar tahan perubahan struktur situs."""
    url = (
        f"https://harga-emas.org/history-harga/"
        f"{date.year}/{BULAN_INDO[date.month]}/{date.strftime('%d')}"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            tables = pd.read_html(StringIO(r.text))

            price_table = None
            for t in tables:
                cols = [str(c).lower() for c in t.columns]
                if any('antam' in c for c in cols):
                    price_table = t
                    break
            if price_table is None:
                log(f"  {date.date()}: tabel Antam tidak ditemukan (percobaan {attempt+1})")
                continue

            antam_col = next(c for c in price_table.columns if 'antam' in str(c).lower())
            satuan_col = price_table.columns[0]

            row = price_table[price_table[satuan_col].astype(str).str.strip() == '1']
            if row.empty:
                numeric_satuan = pd.to_numeric(price_table[satuan_col], errors='coerce')
                row = price_table[numeric_satuan == 1.0]
            if row.empty:
                log(f"  {date.date()}: baris Satuan=1 tidak ditemukan")
                continue

            harga_raw = str(row.iloc[0][antam_col])
            harga = harga_raw.replace("Rp", "").replace(".", "").replace(",", "").strip()
            return float(harga)
        except Exception as e:
            log(f"  {date.date()}: error {type(e).__name__}: {e} (percobaan {attempt+1})")
            time.sleep(2)
    return None


def main():
    if not CSV_PATH.exists():
        log(f"ERROR: {CSV_PATH} tidak ditemukan.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    last_date = df['Date'].max()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    log(f"Data terakhir di CSV : {last_date.date()}")
    log(f"Tanggal hari ini     : {today.date()}")

    if last_date >= today:
        log("Sudah up to date, tidak ada yang perlu discrape.")
        return

    new_rows = []
    current = last_date + timedelta(days=1)
    while current <= today:
        price = scrape_gold_price(current)
        if price:
            log(f"  {current.date()}: OK -> Rp {price:,.0f}")
            new_rows.append({'Date': current, 'Gold Price': price})
        else:
            log(f"  {current.date()}: GAGAL, dilewati")
        current += timedelta(days=1)

    if not new_rows:
        log("Tidak ada data baru yang berhasil discrape. CSV tidak diubah.")
        return

    start_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_str = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        usd_raw = yf.download("USDIDR=X", start=start_str, end=end_str,
                               auto_adjust=True, progress=False)
        usd_raw = usd_raw.reset_index()
        if isinstance(usd_raw.columns, pd.MultiIndex):
            usd_raw.columns = ["_".join(c).strip("_") for c in usd_raw.columns]
        date_col = [c for c in usd_raw.columns if 'Date' in c or 'date' in c][0]
        close_col = [c for c in usd_raw.columns if 'Close' in c or 'close' in c][0]
        usd_new = usd_raw[[date_col, close_col]].copy()
        usd_new.columns = ['Date', 'USDIDR']
        usd_new['Date'] = pd.to_datetime(usd_new['Date']).dt.normalize()
    except Exception as e:
        log(f"Gagal ambil USD/IDR: {e}")
        usd_new = pd.DataFrame(columns=['Date', 'USDIDR'])

    log("Mengambil data GPRD asli dari matteoiacoviello.com ...")
    gpr_lookup = fetch_gpr_daily()
    if gpr_lookup:
        log(f"  Berhasil, {len(gpr_lookup)} baris GPRD ditemukan di file sumber.")
    else:
        log("  Gagal / kosong -> fallback forward-fill nilai GPRD terakhir.")

    last_gpr = {
        'GPRD': df['GPRD'].iloc[-1],
        'GPRD_MA7': df['GPRD_MA7'].iloc[-1],
        'GPRD_MA30': df['GPRD_MA30'].iloc[-1],
    }

    for row in new_rows:
        merged = {'Date': row['Date'], 'Gold Price': row['Gold Price']}
        date_norm = pd.Timestamp(row['Date']).normalize()
        gpr_today = gpr_lookup.get(date_norm)
        for key in ('GPRD', 'GPRD_MA7', 'GPRD_MA30'):
            if gpr_today and gpr_today.get(key) is not None:
                merged[key] = gpr_today[key]
                last_gpr[key] = gpr_today[key]
            else:
                merged[key] = last_gpr[key]
        usd_match = usd_new[usd_new['Date'] == row['Date']]
        merged['USDIDR'] = float(usd_match['USDIDR'].values[0]) \
            if len(usd_match) > 0 else df['USDIDR'].iloc[-1]
        df = pd.concat([df, pd.DataFrame([merged])], ignore_index=True)

    df['Gold Price'] = df['Gold Price'].interpolate(method='linear').ffill().bfill()
    df['USDIDR'] = df['USDIDR'].interpolate(method='linear').ffill().bfill()

    p = df['Gold Price']
    u = df['USDIDR']
    g = df['GPRD']

    df['GoldReturn'] = np.log(p / p.shift(1))
    df['USDReturn'] = np.log(u / u.shift(1))
    df['GoldMA14'] = p.rolling(14).mean()
    df['PriceMA14Ratio'] = p / df['GoldMA14']
    df['RSI14'] = ta.momentum.RSIIndicator(close=p, window=14).rsi()
    df['Vol30'] = df['GoldReturn'].rolling(30).std()
    df['GPRNorm'] = g / (g.rolling(60).mean() + 1e-8)
    df['MonthSin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    df = df.dropna().reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False)
    log(f"Selesai. {len(new_rows)} baris baru ditambahkan. Data terakhir sekarang: {df['Date'].max().date()}")


if __name__ == '__main__':
    main()
