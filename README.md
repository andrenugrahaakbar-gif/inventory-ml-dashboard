# Dashboard Prediksi Permintaan & Manajemen Stok Dinamis (EOQ–ROP–SS)

Sebuah sistem simulasi inventaris berbasis machine learning untuk bisnis retail.
Memprediksi permintaan harian menggunakan model XGBoost, lalu mensimulasikan tingkat stok dengan logika EOQ (Economic Order Quantity), ROP (Reorder Point), dan Safety Stock (Stok Pengaman) secara dinamis.

Demo: https://inventory-ml-dashboard.streamlit.app

# Fitur Utama
Prediksi Permintaan: Model XGBoost dengan fitur lag dan rata-rata bergerak (rolling window)
ROP Dinamis: Titik pemesanan ulang dihitung ulang berdasarkan volatilitas permintaan terkini
Optimasi EOQ: Jumlah pemesanan ekonomis diestimasi dari data permintaan 6 bulan terakhir
Dashboard Interaktif: Jelajahi tren penjualan, simulasi stok harian, dan unduh hasil simulasi
Metrik Bisnis: Total permintaan, frekuensi pemesanan, risiko kehabisan stok (stockout)

# Teknologi yang Digunakan
Machine Learning: XGBoost, Scikit-learn, Joblib
Antarmuka Pengguna: Streamlit, Plotly
Pemrosesan Data: Pandas, NumPy
Deployment: Streamlit Community Cloud
