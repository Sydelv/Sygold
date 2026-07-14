"""
SyGold — Flask web app
Membungkus pipeline prediksi LSTM (identik dengan app.py Streamlit) dan
merender satu halaman HTML statis-terasa dengan data prediksi live.
"""
import pickle
import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
import ta
import yfinance as yf
from flask import Flask, render_template
from tensorflow.keras.models import load_model

from gpr_source import fetch_gpr_daily

app = Flask(__name__)

# ======================================
# KONSTANTA (identik dengan app.py Streamlit)
# ======================================
FEATURES = [
    'GoldReturn', 'USDReturn', 'PriceMA14Ratio',
    'RSI14', 'Vol30', 'GPRNorm', 'MonthSin', 'MonthCos'
]
WINDOW = 30
VOL_FLOOR = 1e-4
BACKTEST_DAYS = 30

BULAN_INDO = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# Cache sederhana di memori (pengganti st.cache_resource / st.cache_data)
_ASSETS = None
_DATA_CACHE = {"df": None, "loaded_at": None}
DATA_TTL_SECONDS = 3600  # 1 jam, sama seperti versi Streamlit


# ======================================
# LOAD MODEL & SCALER (sekali saja saat server start)
# ======================================
def load_assets():
    global _ASSETS
    if _ASSETS is None:
        model_h1 = load_model('model_h1.keras', compile=False)
        model_h7 = load_model('model_h7.keras', compile=False)
        with open('feat_scaler.pkl', 'rb') as f:
            feat_scaler = pickle.load(f)
        with open('y_scaler_h1.pkl', 'rb') as f:
            y_scaler_h1 = pickle.load(f)
        with open('y_scaler_h7.pkl', 'rb') as f:
            y_scaler_h7 = pickle.load(f)
        try:
            with open('shrinkage_w_h1.pkl', 'rb') as f:
                shrinkage_w_h1 = pickle.load(f)
        except FileNotFoundError:
            # Belum di-generate dari notebook -> tanpa shrinkage (murni model)
            shrinkage_w_h1 = 1.0
        _ASSETS = (model_h1, model_h7, feat_scaler, y_scaler_h1, y_scaler_h7, shrinkage_w_h1)
    return _ASSETS


# ======================================
# SCRAPING HARGA EMAS (identik dengan app.py Streamlit)
# ======================================
def scrape_gold_price(date: datetime):
    """
    Ambil harga emas Antam 1 gram dari harga-emas.org.

    Dulu kode ini pakai posisi tetap tables[1].iloc[9,1], tapi itu rapuh —
    kalau situsnya nambah/kurang baris denominasi (mis. baris "2 gram"),
    posisinya geser dan ngambil harga yang salah. Sekarang dicari
    berdasarkan LABEL: baris dengan Satuan == "1" (gram), kolom yang
    namanya mengandung "Antam". Lebih tahan terhadap perubahan struktur.
    """
    url = (
        f"https://harga-emas.org/history-harga/"
        f"{date.year}/{BULAN_INDO[date.month]}/{date.strftime('%d')}"
    )
    for _ in range(3):
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
                return None

            antam_col = next(c for c in price_table.columns if 'antam' in str(c).lower())
            satuan_col = price_table.columns[0]

            row = price_table[price_table[satuan_col].astype(str).str.strip() == '1']
            if row.empty:
                numeric_satuan = pd.to_numeric(price_table[satuan_col], errors='coerce')
                row = price_table[numeric_satuan == 1.0]
            if row.empty:
                return None

            harga_raw = str(row.iloc[0][antam_col])
            harga = harga_raw.replace("Rp", "").replace(".", "").replace(",", "").strip()
            return float(harga)
        except Exception:
            time.sleep(2)
    return None


# ======================================
# LOAD & UPDATE DATA MASTER (identik dengan app.py Streamlit)
# ======================================
def load_and_update_data():
    now = time.time()
    if (_DATA_CACHE["df"] is not None and _DATA_CACHE["loaded_at"] is not None
            and now - _DATA_CACHE["loaded_at"] < DATA_TTL_SECONDS):
        return _DATA_CACHE["df"]

    df = pd.read_csv('master_dataset_featured.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    last_date = df['Date'].max()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    new_rows = []
    current = last_date + timedelta(days=1)
    while current <= today:
        price = scrape_gold_price(current)
        if price:
            new_rows.append({'Date': current, 'Gold Price': price})
        current += timedelta(days=1)

    if new_rows:
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
        except Exception:
            usd_new = pd.DataFrame(columns=['Date', 'USDIDR'])

        # GPR: ambil nilai harian asli dari matteoiacoviello.com kalau tersedia
        # (termasuk GPRD_MA7 & GPRD_MA30 yang sudah dihitung oleh sumbernya),
        # fallback ke forward-fill nilai terakhir kalau file sumbernya belum
        # ter-update atau gagal diakses.
        gpr_lookup = fetch_gpr_daily()
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

    _DATA_CACHE["df"] = df
    _DATA_CACHE["loaded_at"] = now
    return df


# ======================================
# PREDIKSI (identik dengan app.py Streamlit)
# ======================================
def predict(df, feat_scaler, model_h1, model_h7, y_scaler_h1, y_scaler_h7, shrinkage_w_h1=1.0):
    X_all = feat_scaler.transform(df[FEATURES].values)
    prices_all = df['Gold Price'].values
    vol_all = df['Vol30'].values

    last_window = X_all[-WINDOW:]
    X_future = np.expand_dims(last_window, axis=0)
    base_price = prices_all[-1]
    last_vol = max(vol_all[-1], VOL_FLOOR)
    last_date = df['Date'].iloc[-1]

    results = []
    for h in range(1, 8):
        if h == 1:
            pred_scaled = model_h1.predict(X_future, verbose=0)
            pred_volnorm = y_scaler_h1.inverse_transform(pred_scaled)
            pred_ret_model = pred_volnorm.flatten()[0] * last_vol * np.sqrt(1)

            # Shrinkage ke naive (harga tetap): evaluasi skripsi menunjukkan
            # model H+1 murni sedikit kalah dari baseline naive, jadi
            # prediksi di-blend dengan bobot w optimal (dicari dari
            # validation set di notebook, disimpan di shrinkage_w_h1.pkl).
            pred_price_model = base_price * np.exp(pred_ret_model)
            pred_price_shrunk = (
                shrinkage_w_h1 * pred_price_model
                + (1 - shrinkage_w_h1) * base_price
            )
            pred_ret = np.log(pred_price_shrunk / base_price)
        else:
            pred_scaled = model_h7.predict(X_future, verbose=0)
            pred_volnorm = y_scaler_h7.inverse_transform(pred_scaled)
            ret_h1 = results[0]['return']
            ret_h7 = pred_volnorm.flatten()[0] * last_vol * np.sqrt(7)
            pred_ret = ret_h1 + (ret_h7 - ret_h1) * (h - 1) / 6

        pred_price = base_price * np.exp(pred_ret)
        pred_date = last_date + timedelta(days=h)

        results.append({
            'horizon': f'H+{h}',
            'date': pred_date,
            'date_label': pred_date.strftime('%d %b %Y'),
            'price': round(pred_price),
            'price_label': f"Rp {round(pred_price):,.0f}".replace(",", "."),
            'return': pred_ret,
            'change_pct': pred_ret * 100,
        })

    return base_price, last_date, results


# ======================================
# BACKTEST H+1 — replay prediksi "sehari ke depan" untuk N hari terakhir,
# dibandingkan dengan harga aktual yang benar-benar terjadi. Ini beda dari
# `predict()` di atas yang memprediksi ke masa depan (belum ada actual-nya).
# ======================================
def backtest_h1(df, feat_scaler, model_h1, y_scaler_h1, shrinkage_w_h1=1.0, n=BACKTEST_DAYS):
    X_all = feat_scaler.transform(df[FEATURES].values)
    prices_all = df['Gold Price'].values
    vol_all = df['Vol30'].values
    dates_all = df['Date'].values

    total = len(df)
    start_idx = max(WINDOW, total - n)

    results = []
    for i in range(start_idx, total):
        window = X_all[i - WINDOW:i]
        X_input = np.expand_dims(window, axis=0)
        base = prices_all[i - 1]
        vol = max(vol_all[i - 1], VOL_FLOOR)

        pred_scaled = model_h1.predict(X_input, verbose=0)
        pred_volnorm = y_scaler_h1.inverse_transform(pred_scaled)
        pred_ret = pred_volnorm.flatten()[0] * vol * np.sqrt(1)
        pred_price_model = base * np.exp(pred_ret)
        # Shrinkage sama seperti di predict(), supaya grafik akurasi ini
        # mencerminkan prediksi yang sama persis dengan yang ditampilkan
        # ke pengguna, bukan model mentah sebelum shrinkage.
        pred_price = shrinkage_w_h1 * pred_price_model + (1 - shrinkage_w_h1) * base
        actual_price = prices_all[i]

        results.append({
            'date': pd.Timestamp(dates_all[i]),
            'date_label': pd.Timestamp(dates_all[i]).strftime('%d %b'),
            'actual': round(actual_price),
            'predicted': round(pred_price),
            'error_pct': (pred_price - actual_price) / actual_price * 100,
        })

    mape = float(np.mean([abs(r['error_pct']) for r in results])) if results else 0.0
    return results, mape


# ======================================
# ROUTE
# ======================================
@app.route('/')
def index():
    model_h1, model_h7, feat_scaler, y_scaler_h1, y_scaler_h7, shrinkage_w_h1 = load_assets()
    df = load_and_update_data()
    base_price, last_date, results = predict(
        df, feat_scaler, model_h1, model_h7, y_scaler_h1, y_scaler_h7, shrinkage_w_h1
    )

    r1 = results[0]
    r7 = results[6]

    hist_30 = df.tail(30)[['Date', 'Gold Price']].copy()
    chart_labels = [d.strftime('%d %b') for d in hist_30['Date']] + [r['date_label'] for r in results]
    chart_actual = [round(v) for v in hist_30['Gold Price']] + [None] * len(results)
    chart_pred = [None] * (len(hist_30) - 1) + [round(base_price)] + [r['price'] for r in results]

    backtest, backtest_mape = backtest_h1(df, feat_scaler, model_h1, y_scaler_h1, shrinkage_w_h1)
    backtest_labels = [b['date_label'] for b in backtest]
    backtest_actual = [b['actual'] for b in backtest]
    backtest_pred = [b['predicted'] for b in backtest]

    return render_template(
        'index.html',
        base_price=base_price,
        base_price_label=f"Rp {round(base_price):,.0f}".replace(",", "."),
        last_date_label=last_date.strftime('%A, %d %B %Y'),
        last_update_label=df['Date'].max().strftime('%d %B %Y'),
        n_rows=f"{len(df):,}".replace(",", "."),
        r1=r1,
        r7=r7,
        results=results,
        chart_labels=chart_labels,
        chart_actual=chart_actual,
        chart_pred=chart_pred,
        backtest=backtest,
        backtest_mape=backtest_mape,
        backtest_labels=backtest_labels,
        backtest_actual=backtest_actual,
        backtest_pred=backtest_pred,
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
