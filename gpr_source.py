"""
Ambil data GPRD (Geopolitical Risk Daily) langsung dari sumber resminya:
Caldara & Iacoviello, https://www.matteoiacoviello.com/gpr.htm

File di-update tiap hari Senin oleh pemiliknya (bukan bulanan seperti
asumsi awal). Kalau download/parsing gagal (mis. situs down, format
berubah), fetch_gpr_daily() balikin dict kosong {} supaya caller bisa
fallback ke forward-fill nilai terakhir yang ada di CSV — jadi pipeline
tetap jalan walau sumber eksternal ini bermasalah.

Catatan struktur file (dikonfirmasi 2026-07): sheet-nya punya kolom
'DAY' (integer YYYYMMDD, JANGAN dipakai sebagai tanggal) dan kolom
'date' (datetime asli, ini yang benar dipakai). Kolom GPRD_MA7 dan
GPRD_MA30 sudah tersedia langsung dari sumber, jadi tidak perlu
dihitung ulang secara manual.
"""
from io import BytesIO

import pandas as pd
import requests

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def fetch_gpr_daily():
    """
    Return: dict {pd.Timestamp (normalized): {'GPRD': float, 'GPRD_MA7': float|None,
    'GPRD_MA30': float|None}} atau {} kalau gagal.
    """
    try:
        r = requests.get(GPR_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()

        raw = pd.read_excel(BytesIO(r.content), engine='xlrd')
        raw.columns = [str(c).strip() for c in raw.columns]

        # Prioritaskan kolom 'date' (datetime asli) di atas 'DAY' (integer
        # YYYYMMDD) yang formatnya beda dan gampang salah parse.
        if 'date' in raw.columns:
            date_col = 'date'
        elif 'Date' in raw.columns:
            date_col = 'Date'
        else:
            date_col = next((c for c in raw.columns if 'date' in c.lower()), None)

        gprd_col = next((c for c in raw.columns if c.upper() == 'GPRD'), None)
        if gprd_col is None:
            return {}
        if date_col is None:
            return {}

        raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
        raw = raw.dropna(subset=[date_col, gprd_col])

        has_ma7 = 'GPRD_MA7' in raw.columns
        has_ma30 = 'GPRD_MA30' in raw.columns

        lookup = {}
        for _, row in raw.iterrows():
            d = pd.Timestamp(row[date_col]).normalize()
            lookup[d] = {
                'GPRD': float(row[gprd_col]),
                'GPRD_MA7': float(row['GPRD_MA7']) if has_ma7 and pd.notna(row['GPRD_MA7']) else None,
                'GPRD_MA30': float(row['GPRD_MA30']) if has_ma30 and pd.notna(row['GPRD_MA30']) else None,
            }
        return lookup
    except Exception:
        return {}
