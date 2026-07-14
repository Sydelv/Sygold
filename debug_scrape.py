"""
Diagnostic: cek apakah scrape_gold_price() masih bisa narik data
dari harga-emas.org. Jalankan: python debug_scrape.py
"""
from datetime import datetime, timedelta
from io import StringIO
import requests
import pandas as pd

BULAN_INDO = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

def scrape_gold_price_debug(date: datetime):
    url = (
        f"https://harga-emas.org/history-harga/"
        f"{date.year}/{BULAN_INDO[date.month]}/{date.strftime('%d')}"
    )
    print(f"\n[TEST] Tanggal: {date.strftime('%Y-%m-%d')}")
    print(f"[TEST] URL     : {url}")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        print(f"[TEST] Status  : {r.status_code}")
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        print(f"[TEST] Jumlah tabel ditemukan: {len(tables)}")
        if len(tables) < 2:
            print("[TEST] GAGAL: tabel ke-2 tidak ditemukan (struktur halaman mungkin berubah)")
            return None
        print("[TEST] Preview tabel[1]:")
        print(tables[1].head(12))
        harga_raw = str(tables[1].iloc[9, 1])
        print(f"[TEST] Raw cell [9,1]: {harga_raw!r}")
        harga = harga_raw.replace("Rp", "").replace(".", "").replace(",", "").strip()
        return float(harga)
    except Exception as e:
        print(f"[TEST] ERROR: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    # Coba beberapa tanggal terakhir
    today = datetime.now()
    print(f"Tanggal sistem komputer kamu sekarang: {today.strftime('%Y-%m-%d')}")
    for i in range(1, 6):
        d = today - timedelta(days=i)
        result = scrape_gold_price_debug(d)
        print(f"[TEST] >>> Hasil harga: {result}")
