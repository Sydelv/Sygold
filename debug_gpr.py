"""
Diagnostic: cek apakah fetch_gpr_daily() berhasil download & parse
file GPRD dari matteoiacoviello.com. Jalankan: python debug_gpr.py
"""
import requests
import pandas as pd
from io import BytesIO

from gpr_source import GPR_URL, fetch_gpr_daily

print(f"URL   : {GPR_URL}")

r = requests.get(GPR_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
print(f"Status: {r.status_code}")
print(f"Size  : {len(r.content)} bytes")

raw = pd.read_excel(BytesIO(r.content), engine='xlrd')
raw.columns = [str(c).strip() for c in raw.columns]
print(f"\nKolom yang ditemukan: {list(raw.columns)}")
print(f"\n5 baris pertama:")
print(raw.head())
print(f"\n5 baris terakhir:")
print(raw.tail())

print("\n--- Hasil fetch_gpr_daily() ---")
lookup = fetch_gpr_daily()
print(f"Jumlah tanggal ter-parse: {len(lookup)}")
if lookup:
    latest = max(lookup.keys())
    print(f"Tanggal terbaru di file: {latest.date()} -> GPRD = {lookup[latest]}")
else:
    print("KOSONG — cek apakah nama kolom di atas cocok dengan yang dicari di gpr_source.py")
