import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta, date  

st.set_page_config(
    page_title="Inventory Management System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
# LANGUAGE DICTIONARY
# ══════════════════════════════════════════════════════════════════════
LANG = {
    "EN": {
        # Navigation
        "nav_overview"       : "📊 Overview",
        "nav_simulation"     : "📦 Inventory Simulation",
        "navigation"         : "Navigation",
        "language_label"     : "🌐 Language",

        # Page 1 – Sales Overview
        "page1_title"        : "📊 Sales Overview & Business Insights",
        "start_date"         : "Start Date",
        "end_date"           : "End Date",
        "err_date_order"     : "Start date must be before end date.",
        "kpi_total_sales"    : "Total Sales",
        "kpi_products"       : "Products",
        "kpi_period"         : "Period",
        "warn_no_data"       : "No data in selected date range.",
        "top_bottom"         : "🏆 Top & Bottom Products",
        "top5"               : "**Top 5**",
        "bottom5"            : "**Bottom 5**",
        "filter_trend"       : "🔍 Filter Sales Trend",
        "filter_by"          : "Filter by:",
        "filter_none"        : "None",
        "filter_product"     : "Product",
        "filter_category"    : "Category",
        "select_product"     : "Select Product",
        "selected_label"     : "**Selected:**",
        "select_category"    : "Select Category",
        "warn_no_kategori"   : "Column 'KATEGORI' not found.",
        "sales_trend"        : "📈 Sales Trend",
        "view_as"            : "View as:",
        "daily"              : "Daily",
        "monthly"            : "Monthly",
        "daily_sales"        : "Daily Sales",
        "monthly_sales"      : "Monthly Sales",
        "demand_dist"        : "📊 Demand Distribution",
        "top10_title"        : "🥇 Top 10 Best-Selling Products",
        "top10_chart_title"  : "Top 10 by Sales Volume",
        "cat_pie_title"      : "🍰 Sales by Product Category",
        "cat_pie_chart"      : "Sales Distribution by Category",

        # Page 2 – Inventory Simulation
        "page2_title"        : "📦 Inventory Simulation: EOQ & ROP",
        "select_product_code": "Select Product Code",
        "product_label"      : "**Product:**",
        "tab_eoq"            : "🔁 EOQ Calculator",
        "tab_rop"            : "⚠️ ROP Simulation",
        "eoq_subtitle"       : "Economic Order Quantity (EOQ) – Auto Estimated",
        "est_monthly_demand" : "**Estimated Monthly Demand:** {val:,} units (from last 6 months)",
        "ordering_cost"      : "Ordering Cost per Order (IDR)",
        "holding_cost"       : "Holding Cost per Unit per Month (IDR)",
        "warn_eoq"           : "Unable to calculate EOQ (check input values).",
        "eoq_result"         : "✅ **Recommended EOQ:** {val:,} units",
        "rop_subtitle"       : "ROP Simulation with EOQ Restocking",
        "custom_end_date"    : "Custom end date",
        "err_end_date"       : "End date must be after start date.",
        "sim_period"         : "Simulation Period",
        "period_1m"          : "1 Month",
        "period_3m"          : "3 Months",
        "period_6m"          : "6 Months",
        "lead_time"          : "Lead Time (days)",
        "initial_stock"      : "Initial Stock",
        "restock_info"       : "Restocking quantity: **{val} units (EOQ)**",
        "btn_run"            : "🚀 Run Simulation",
        "key_params"         : "⚙️ Key Parameters",
        "service_level"      : "Service Level",
        "std_dev"            : "Std Dev",
        "z_score"            : "Z-Score",
        "sim_summary"        : "📋 Simulation Summary",
        "total_forecast"     : "Total Forecast Demand",
        "avg_daily"          : "Avg Daily Demand",
        "orders_placed"      : "Orders Placed",
        "sim_results"        : "📅 Simulation Results",
        "inventory_chart"    : "📉 Inventory Simulation Over Time",
        "demand_chart"       : "📈 Predicted Demand Only",
        "demand_chart_title" : "Predicted Daily Demand",
        "btn_download"       : "📥 Download Results",
        "err_model_not_found": "❌ Model file not found: {e}",
        "info_run_deploy"    : "Make sure Deploy_model_FIXED.py has been run and the saved_models/ folder contains the appropriate model files.",
        "err_sim_failed"     : "Simulation failed: {e}",

        # Status labels
        "status_sufficient"  : "Sufficient",
        "status_reorder"     : "Reorder Required",
        "status_awaiting"    : "Awaiting Order Arrival",
        "status_no_data"     : "Not enough historical data",

        # Footer
        "footer"             : "© 2025 Andre Nugraha. All rights reserved.",
    },
    "ID": {
        # Navigation
        "nav_overview"       : "📊 Overview",
        "nav_simulation"     : "📦 Simulasi Inventori",
        "navigation"         : "Navigasi",
        "language_label"     : "🌐 Bahasa",

        # Page 1 – Sales Overview
        "page1_title"        : "📊 Overview Penjualan & Insight Bisnis",
        "start_date"         : "Tanggal Mulai",
        "end_date"           : "Tanggal Akhir",
        "err_date_order"     : "Tanggal mulai harus sebelum tanggal akhir.",
        "kpi_total_sales"    : "Total Penjualan",
        "kpi_products"       : "Produk",
        "kpi_period"         : "Periode",
        "warn_no_data"       : "Tidak ada data pada rentang tanggal yang dipilih.",
        "top_bottom"         : "🏆 Produk Terlaris & Terbawah",
        "top5"               : "**Top 5**",
        "bottom5"            : "**Bottom 5**",
        "filter_trend"       : "🔍 Filter Tren Penjualan",
        "filter_by"          : "Filter berdasarkan:",
        "filter_none"        : "Semua",
        "filter_product"     : "Produk",
        "filter_category"    : "Kategori",
        "select_product"     : "Pilih Produk",
        "selected_label"     : "**Dipilih:**",
        "select_category"    : "Pilih Kategori",
        "warn_no_kategori"   : "Kolom 'KATEGORI' tidak ditemukan.",
        "sales_trend"        : "📈 Tren Penjualan",
        "view_as"            : "Tampilkan sebagai:",
        "daily"              : "Harian",
        "monthly"            : "Bulanan",
        "daily_sales"        : "Penjualan Harian",
        "monthly_sales"      : "Penjualan Bulanan",
        "demand_dist"        : "📊 Distribusi Permintaan",
        "top10_title"        : "🥇 Top 10 Produk Terlaris",
        "top10_chart_title"  : "Top 10 Berdasarkan Volume Penjualan",
        "cat_pie_title"      : "🍰 Penjualan per Kategori Produk",
        "cat_pie_chart"      : "Distribusi Penjualan per Kategori",

        # Page 2 – Inventory Simulation
        "page2_title"        : "📦 Simulasi Inventori: EOQ & ROP",
        "select_product_code": "Pilih Kode Produk",
        "product_label"      : "**Produk:**",
        "tab_eoq"            : "🔁 Kalkulator EOQ",
        "tab_rop"            : "⚠️ Simulasi ROP",
        "eoq_subtitle"       : "Economic Order Quantity (EOQ) – Estimasi Otomatis",
        "est_monthly_demand" : "**Estimasi Permintaan Bulanan:** {val:,} unit (dari 6 bulan terakhir)",
        "ordering_cost"      : "Biaya Pemesanan per Order (IDR)",
        "holding_cost"       : "Biaya Penyimpanan per Unit per Bulan (IDR)",
        "warn_eoq"           : "EOQ tidak dapat dihitung (periksa nilai input).",
        "eoq_result"         : "✅ **EOQ yang Disarankan:** {val:,} unit",
        "rop_subtitle"       : "Simulasi ROP dengan Restok EOQ",
        "custom_end_date"    : "Tanggal akhir kustom",
        "err_end_date"       : "Tanggal akhir harus setelah tanggal mulai.",
        "sim_period"         : "Periode Simulasi",
        "period_1m"          : "1 Bulan",
        "period_3m"          : "3 Bulan",
        "period_6m"          : "6 Bulan",
        "lead_time"          : "Lead Time (hari)",
        "initial_stock"      : "Stok Awal",
        "restock_info"       : "Jumlah restok: **{val} unit (EOQ)**",
        "btn_run"            : "🚀 Jalankan Simulasi",
        "key_params"         : "⚙️ Parameter Utama",
        "service_level"      : "Service Level",
        "std_dev"            : "Std Deviasi",
        "z_score"            : "Z-Score",
        "sim_summary"        : "📋 Ringkasan Simulasi",
        "total_forecast"     : "Total Perkiraan Permintaan",
        "avg_daily"          : "Rata-rata Permintaan Harian",
        "orders_placed"      : "Pesanan Dibuat",
        "sim_results"        : "📅 Hasil Simulasi",
        "inventory_chart"    : "📉 Simulasi Inventori dari Waktu ke Waktu",
        "demand_chart"       : "📈 Hanya Perkiraan Permintaan",
        "demand_chart_title" : "Perkiraan Permintaan Harian",
        "btn_download"       : "📥 Unduh Hasil",
        "err_model_not_found": "❌ File model tidak ditemukan: {e}",
        "info_run_deploy"    : "Pastikan Deploy_model_FIXED.py sudah dijalankan dan folder saved_models/ berisi file model yang sesuai.",
        "err_sim_failed"     : "Simulasi gagal: {e}",

        # Status labels
        "status_sufficient"  : "Stok Cukup",
        "status_reorder"     : "Perlu Reorder",
        "status_awaiting"    : "Menunggu Kedatangan Pesanan",
        "status_no_data"     : "Data historis tidak cukup",

        # Footer
        "footer"             : "© 2025 Andre Nugraha. Hak cipta dilindungi.",
    },
}

# ── Path ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
DATA_DIR   = BASE / "Data"
MODELS_DIR = BASE / "saved_models"

# ══════════════════════════════════════════════════════════════════════
# LANGUAGE SELECTOR (Sidebar – always on top)
# ══════════════════════════════════════════════════════════════════════
if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"

with st.sidebar:
    col_flag_en, col_flag_id = st.columns(2)
    with col_flag_en:
        if st.button("🇬🇧 English", use_container_width=True,
                     type="primary" if st.session_state["lang"] == "EN" else "secondary"):
            st.session_state["lang"] = "EN"
            st.rerun()
    with col_flag_id:
        if st.button("🇮🇩 Indonesia", use_container_width=True,
                     type="primary" if st.session_state["lang"] == "ID" else "secondary"):
            st.session_state["lang"] = "ID"
            st.rerun()

    st.divider()

# Shortcut helper
L = LANG[st.session_state["lang"]]

# ── Cache / Load ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    model_files = list(MODELS_DIR.glob("final_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            f"Tidak ada file 'final_model_*.joblib' di {MODELS_DIR}. "
            "Jalankan Deploy_model_FIXED.py terlebih dahulu."
        )
    model_path = sorted(model_files)[-1]
    model = joblib.load(model_path)

    scaler       = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_list = joblib.load(MODELS_DIR / "feature_list.joblib")

    encoders_path = MODELS_DIR / "label_encoders.joblib"
    if encoders_path.exists():
        label_encoders = joblib.load(encoders_path)
    else:
        le_kode = joblib.load(MODELS_DIR / "label_encoder.joblib")
        label_encoders = {'KODE': le_kode}

    return model, scaler, label_encoders, feature_list

@st.cache_data
def load_data():
    try:
        data_path = DATA_DIR / "rekap_penjualan.xlsx"
        data = pd.read_excel(data_path)
        data['Tanggal'] = pd.to_datetime(data['Tanggal'])
        return data
    except Exception:
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Tanggal'    : dates,
            'KODE'       : np.random.choice([1, 2, 3], len(dates)),
            'NAMA BARANG': np.random.choice(['Product A', 'Product B', 'Product C'], len(dates)),
            'QTY'        : np.random.randint(10, 100, len(dates)),
            'KATEGORI'   : np.random.choice(['Category 1', 'Category 2'], len(dates)),
            'SUPPLIER'   : np.random.choice(['Supplier A', 'Supplier B'], len(dates)),
        })
        return sample_data

def safe_encode(label_encoders: dict, col: str, value) -> int:
    if col not in label_encoders:
        return 0
    enc = label_encoders[col]
    try:
        return int(enc.transform([str(value)])[0])
    except Exception:
        return 0

def calculate_eoq(demand_monthly, ordering_cost, holding_cost_per_unit):
    if holding_cost_per_unit <= 0 or ordering_cost <= 0 or demand_monthly <= 0:
        return 0
    eoq = np.sqrt((2 * demand_monthly * ordering_cost) / holding_cost_per_unit)
    return max(1, round(eoq))

def calculate_safety_stock_fmcg(daily_demand_series, lead_time_days, service_level, lead_time_std=None):
    if len(daily_demand_series) < 7:
        return 0, np.mean(daily_demand_series) if len(daily_demand_series) > 0 else 0, 0, 0
    avg_daily_demand = np.mean(daily_demand_series)
    std_daily_demand = np.std(daily_demand_series)
    try:
        z_score = stats.norm.ppf(service_level)
    except:
        z_score = 1.645
    if lead_time_std is not None and lead_time_std > 0:
        safety_stock = z_score * np.sqrt(
            (lead_time_days * (std_daily_demand ** 2)) +
            ((avg_daily_demand ** 2) * (lead_time_std ** 2))
        )
    else:
        safety_stock = z_score * std_daily_demand * np.sqrt(lead_time_days)
    return safety_stock, avg_daily_demand, std_daily_demand, z_score

def calculate_rop_ss(daily_demand_series, lead_time_days, service_level):
    safety_stock, avg_daily_demand, std_daily_demand, z_score = calculate_safety_stock_fmcg(
        daily_demand_series, lead_time_days, service_level
    )
    rop = (avg_daily_demand * lead_time_days) + safety_stock
    return {
        'rop'              : rop,
        'safety_stock'     : safety_stock,
        'avg_daily_demand' : avg_daily_demand,
        'std_daily_demand' : std_daily_demand,
        'z_score'          : z_score,
        'lead_time_demand' : avg_daily_demand * lead_time_days
    }

def estimate_monthly_demand(data, product_code, months_back=6):
    df = data[data['KODE'] == product_code].copy()
    df = df.sort_values('Tanggal')
    if df.empty:
        return 0
    end_date   = df['Tanggal'].max()
    start_date = end_date - pd.DateOffset(months=months_back)
    df_recent  = df[df['Tanggal'] >= start_date]
    if df_recent.empty:
        df_recent = df
    total_days   = (df_recent['Tanggal'].max() - df_recent['Tanggal'].min()).days
    if total_days == 0:
        total_days = 30
    total_months   = max(total_days / 30.0, 1.0)
    total_qty      = df_recent['QTY'].sum()
    monthly_demand = total_qty / total_months
    return max(1, round(monthly_demand))

# ══════════════════════════════════════════════════════════════════════
# Prediction Function
# ══════════════════════════════════════════════════════════════════════
def predict_demand_rop(
    data, product_code, product_name,
    start_date, period_days,
    lead_time, initial_stock,
    model, scaler, label_encoders, feature_list, eoq
):
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)

    prediction_dates = [start_date + timedelta(days=i) for i in range(period_days)]

    data = data.copy()
    data['Tanggal'] = pd.to_datetime(data['Tanggal'])

    df_code = data[data['KODE'] == product_code].copy()
    df_code = df_code.sort_values('Tanggal').reset_index(drop=True)

    if not df_code.empty and 'KATEGORI' in df_code.columns:
        kategori_raw = df_code['KATEGORI'].iloc[0]
        supplier_raw = df_code['SUPPLIER'].iloc[0] if 'SUPPLIER' in df_code.columns else ''
    else:
        kategori_raw = ''
        supplier_raw = ''

    kode_enc     = safe_encode(label_encoders, 'KODE',     product_code)
    kategori_enc = safe_encode(label_encoders, 'KATEGORI', kategori_raw)
    supplier_enc = safe_encode(label_encoders, 'SUPPLIER', supplier_raw)

    all_features     = feature_list['all_features']
    top_features     = feature_list['top_features']
    numeric_features = feature_list.get(
        'numeric_features',
        ['Lag_1_days', 'Lag_2_days', 'Lag_7_days',
         'Rolling_Mean_3', 'Rolling_Mean_7',
         'Rolling_Std_3', 'Rolling_Std_7']
    )

    results         = []
    remaining_stock = initial_stock
    order_schedule  = {}
    order_frequency = 0
    rop_info        = calculate_rop_ss([1.0] * 7, lead_time, service_level=0.95)

    # Use translated status labels
    STATUS_SUFFICIENT = L["status_sufficient"]
    STATUS_REORDER    = L["status_reorder"]
    STATUS_AWAITING   = L["status_awaiting"]
    STATUS_NO_DATA    = L["status_no_data"]

    for date_pred in prediction_dates:
        date_pred = pd.Timestamp(date_pred)

        if date_pred in order_schedule:
            remaining_stock += order_schedule[date_pred]
            del order_schedule[date_pred]

        df_prev = df_code[df_code['Tanggal'] < date_pred].copy()

        if len(df_prev) < 7:
            results.append({
                'Date'           : date_pred.date(),
                'Product_Name'   : product_name,
                'Predicted_QTY'  : 0,
                'ROP'            : 0,
                'Safety_Stock'   : 0,
                'Remaining_Stock': round(remaining_stock),
                'Status'         : STATUS_NO_DATA,
                'Order_Note'     : '',
            })
            continue

        qty_hist = df_prev['QTY'].values.astype(float)

        lag_1          = qty_hist[-1]
        lag_2          = qty_hist[-2]
        lag_7          = qty_hist[-7] if len(qty_hist) >= 7 else qty_hist[0]
        rolling_mean_3 = float(np.mean(qty_hist[-3:]))
        rolling_mean_7 = float(np.mean(qty_hist[-7:]))
        rolling_std_3  = float(np.std(qty_hist[-3:]))  if len(qty_hist) >= 3 else 0.0
        rolling_std_7  = float(np.std(qty_hist[-7:]))  if len(qty_hist) >= 7 else 0.0

        features_pred = pd.DataFrame([{
            'KODE'          : kode_enc,
            'KATEGORI'      : kategori_enc,
            'SUPPLIER'      : supplier_enc,
            'Month'         : date_pred.month,
            'Day'           : date_pred.day,
            'DayOfWeek'     : date_pred.dayofweek,
            'Lag_1_days'    : lag_1,
            'Lag_2_days'    : lag_2,
            'Lag_7_days'    : lag_7,
            'Rolling_Mean_3': rolling_mean_3,
            'Rolling_Mean_7': rolling_mean_7,
            'Rolling_Std_3' : rolling_std_3,
            'Rolling_Std_7' : rolling_std_7,
        }])

        num_cols_available   = [c for c in numeric_features if c in features_pred.columns]
        features_pred_scaled = features_pred.copy()
        features_pred_scaled[num_cols_available] = scaler.transform(
            features_pred[num_cols_available].values
        )

        top_available  = [f for f in top_features if f in features_pred_scaled.columns]
        features_input = features_pred_scaled[top_available]

        pred_qty = float(model.predict(features_input.values)[0])
        pred_qty = max(0.0, pred_qty)

        daily_history = qty_hist[-180:]
        rop_info      = calculate_rop_ss(daily_history, lead_time, service_level=0.95)
        rop           = rop_info['rop']
        safety_stock  = rop_info['safety_stock']

        remaining_stock = max(0.0, remaining_stock - pred_qty)

        status        = STATUS_SUFFICIENT
        order_note    = ""
        pending_order = any(t > date_pred for t in order_schedule.keys())

        if remaining_stock <= rop and not pending_order:
            arrival_date              = date_pred + pd.Timedelta(days=lead_time)
            order_schedule[arrival_date] = eoq
            order_note                = f"Order {eoq} units (arrives {arrival_date.date()})"
            order_frequency          += 1
            status                    = STATUS_REORDER
        elif pending_order:
            status     = STATUS_AWAITING
            order_note = "Order in transit"

        results.append({
            'Date'           : date_pred.date(),
            'Product_Name'   : product_name,
            'Predicted_QTY'  : round(pred_qty),
            'ROP'            : round(rop),
            'Safety_Stock'   : round(safety_stock),
            'Remaining_Stock': round(remaining_stock),
            'Status'         : status,
            'Order_Note'     : order_note,
        })

        new_row = pd.DataFrame([{
            'Tanggal'    : date_pred,
            'KODE'       : product_code,
            'NAMA BARANG': product_name,
            'QTY'        : pred_qty,
        }])
        df_code = pd.concat([df_code, new_row], ignore_index=True)
        df_code = df_code.sort_values('Tanggal').reset_index(drop=True)

    result_df = pd.DataFrame(results)
    result_df.attrs['order_frequency'] = order_frequency
    return result_df, rop_info

# ══════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════
data      = load_data()
code_list = sorted(data['KODE'].unique().tolist())

# Navigation (using translated labels)
with st.sidebar:
    page = st.radio(
        L["navigation"],
        [L["nav_overview"], L["nav_simulation"]]
    )

# ════════════════════════════════
# PAGE 1 — Sales Overview
# ════════════════════════════════
if page == L["nav_overview"]:
    st.title(L["page1_title"])

    min_date = data['Tanggal'].min().date()
    max_date = data['Tanggal'].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_filter = st.date_input(L["start_date"], min_date,
                                     min_value=min_date, max_value=max_date)
    with col2:
        end_filter = st.date_input(L["end_date"], max_date,
                                   min_value=min_date, max_value=max_date)

    if start_filter > end_filter:
        st.error(L["err_date_order"])
        st.stop()

    filtered_data = data[
        (data['Tanggal'].dt.date >= start_filter) &
        (data['Tanggal'].dt.date <= end_filter)
    ].copy()

    # KPIs
    total_sales     = filtered_data['QTY'].sum()
    unique_products = filtered_data['KODE'].nunique()

    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.metric(L["kpi_total_sales"], f"{total_sales:,.0f}")
    col_k2.metric(L["kpi_products"],    unique_products)
    col_k3.metric(L["kpi_period"],      f"{start_filter} → {end_filter}")

    if filtered_data.empty:
        st.warning(L["warn_no_data"])
        st.stop()

    # Top & Bottom products
    prod_agg = (
        filtered_data
        .groupby('KODE')
        .agg(Total_Qty=('QTY', 'sum'), Product_Name=('NAMA BARANG', 'first'))
        .reset_index()
        .sort_values('Total_Qty', ascending=False)
    )

    st.subheader(L["top_bottom"])
    ca, cb = st.columns(2)
    with ca:
        st.write(L["top5"])
        st.dataframe(prod_agg.head(5)[['Product_Name', 'Total_Qty']],
                     use_container_width=True)
    with cb:
        st.write(L["bottom5"])
        st.dataframe(prod_agg.tail(5)[['Product_Name', 'Total_Qty']],
                     use_container_width=True)

    st.subheader(L["filter_trend"])
    filter_type = st.radio(
        L["filter_by"],
        [L["filter_none"], L["filter_product"], L["filter_category"]],
        horizontal=True
    )
    visual_data = filtered_data.copy()

    if filter_type == L["filter_product"]:
        selected_code = st.selectbox(L["select_product"], sorted(filtered_data['KODE'].unique()))
        visual_data   = filtered_data[filtered_data['KODE'] == selected_code]
        pname         = visual_data['NAMA BARANG'].iloc[0] if not visual_data.empty else "Unknown"
        st.markdown(f"{L['selected_label']} `{selected_code}` — **{pname}**")
    elif filter_type == L["filter_category"]:
        if 'KATEGORI' in filtered_data.columns:
            selected_cat = st.selectbox(L["select_category"],
                                        sorted(filtered_data['KATEGORI'].dropna().unique()))
            visual_data  = filtered_data[filtered_data['KATEGORI'] == selected_cat]
        else:
            st.warning(L["warn_no_kategori"])

    st.subheader(L["sales_trend"])
    view_opt = st.radio(L["view_as"], [L["daily"], L["monthly"]], horizontal=True)

    if view_opt == L["daily"]:
        daily = visual_data.groupby('Tanggal')['QTY'].sum().reset_index()
        fig   = px.line(daily, x='Tanggal', y='QTY', title=L["daily_sales"]) \
                if not daily.empty \
                else go.Figure().update_layout(title=L["daily_sales"])
    else:
        temp = visual_data.copy()
        temp['MonthYear'] = temp['Tanggal'].dt.to_period('M')
        monthly = temp.groupby('MonthYear')['QTY'].sum().reset_index()
        monthly['MonthYear'] = pd.to_datetime(monthly['MonthYear'].astype(str))
        fig = px.line(monthly, x='MonthYear', y='QTY', title=L["monthly_sales"]) \
              if not monthly.empty \
              else go.Figure().update_layout(title=L["monthly_sales"])
        fig.update_xaxes(tickformat="%b %Y", dtick="M1")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader(L["demand_dist"])
    st.plotly_chart(px.histogram(filtered_data, x='QTY', nbins=50),
                    use_container_width=True)

    st.subheader(L["top10_title"])
    top10 = prod_agg.head(10)
    if not top10.empty:
        fig_top10 = px.bar(top10, x='Total_Qty', y='Product_Name',
                           orientation='h', title=L["top10_chart_title"])
        fig_top10.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top10, use_container_width=True)

    if 'KATEGORI' in filtered_data.columns:
        st.subheader(L["cat_pie_title"])
        cat_sales = filtered_data.groupby('KATEGORI')['QTY'].sum().reset_index()
        if not cat_sales.empty:
            st.plotly_chart(
                px.pie(cat_sales, values='QTY', names='KATEGORI',
                       title=L["cat_pie_chart"]),
                use_container_width=True
            )

# ════════════════════════════════
# PAGE 2 — Inventory Simulation
# ════════════════════════════════
elif page == L["nav_simulation"]:
    st.title(L["page2_title"])

    product_code = st.selectbox(L["select_product_code"], code_list)
    product_name = data.loc[data['KODE'] == product_code, 'NAMA BARANG'].values[0]
    st.markdown(f"{L['product_label']} {product_name}")

    tab_eoq, tab_rop = st.tabs([L["tab_eoq"], L["tab_rop"]])

    # ── EOQ Tab ───────────────────────────────────────────────────────────────
    with tab_eoq:
        st.subheader(L["eoq_subtitle"])

        monthly_demand_est = estimate_monthly_demand(data, product_code, months_back=6)

        col_a, col_b = st.columns(2)
        with col_a:
            st.info(L["est_monthly_demand"].format(val=monthly_demand_est))

        ordering_cost = st.number_input(L["ordering_cost"], min_value=1, value=50000)
        holding_cost  = st.number_input(L["holding_cost"],  min_value=1, value=200)

        eoq = calculate_eoq(monthly_demand_est, ordering_cost, holding_cost)
        if eoq == 0:
            st.warning(L["warn_eoq"])
        else:
            st.success(L["eoq_result"].format(val=eoq))

        st.session_state['eoq_value']      = eoq
        st.session_state['monthly_demand'] = monthly_demand_est

    # ── ROP Tab ───────────────────────────────────────────────────────────────
    with tab_rop:
        st.subheader(L["rop_subtitle"])

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(L["start_date"], datetime.today().date())
        with col2:
            use_custom = st.checkbox(L["custom_end_date"])
            if use_custom:
                end_date = st.date_input(L["end_date"], start_date + timedelta(days=30))
                if end_date <= start_date:
                    st.error(L["err_end_date"])
                    period_days = 1
                else:
                    period_days = (end_date - start_date).days + 1
            else:
                period_map = {
                    L["period_1m"]: 30,
                    L["period_3m"]: 90,
                    L["period_6m"]: 180,
                }
                period_opt  = st.selectbox(L["sim_period"], list(period_map.keys()))
                period_days = period_map[period_opt]

        col3, col4 = st.columns(2)
        with col3:
            lead_time     = st.number_input(L["lead_time"],     min_value=1, value=3)
        with col4:
            initial_stock = st.number_input(L["initial_stock"], min_value=0, value=100)

        eoq_value = st.session_state.get('eoq_value', 100)
        st.info(L["restock_info"].format(val=eoq_value))

        if st.button(L["btn_run"]):
            try:
                model, scaler, label_encoders, feature_list = load_components()

                result_df, rop_info = predict_demand_rop(
                    data           = data,
                    product_code   = product_code,
                    product_name   = product_name,
                    start_date     = start_date,
                    period_days    = period_days,
                    lead_time      = lead_time,
                    initial_stock  = initial_stock,
                    model          = model,
                    scaler         = scaler,
                    label_encoders = label_encoders,
                    feature_list   = feature_list,
                    eoq            = eoq_value,
                )

                # Key Parameters
                st.subheader(L["key_params"])
                p1, p2, p3 = st.columns(3)
                p1.metric(L["service_level"], "95%")
                p2.metric(L["std_dev"],       f"{rop_info['std_daily_demand']:.2f}")
                p3.metric(L["z_score"],       f"{rop_info['z_score']:.2f}")

                # Summary
                st.subheader(L["sim_summary"])
                total_demand  = result_df['Predicted_QTY'].sum()
                avg_demand    = result_df['Predicted_QTY'].mean()
                orders_placed = result_df.attrs.get('order_frequency', 0)

                s1, s2, s3 = st.columns(3)
                s1.metric(L["total_forecast"], f"{total_demand:.0f}")
                s2.metric(L["avg_daily"],      f"{avg_demand:.1f}")
                s3.metric(L["orders_placed"],  orders_placed)

                # Results table
                st.subheader(L["sim_results"])
                st.dataframe(result_df, use_container_width=True)

                st.subheader(L["inventory_chart"])
                if not result_df.empty:
                    chart_data = result_df.set_index('Date')[
                        ['Predicted_QTY', 'ROP', 'Remaining_Stock']
                    ]
                    st.line_chart(chart_data)

                st.subheader(L["demand_chart"])
                if not result_df.empty:
                    st.plotly_chart(
                        px.line(result_df, x='Date', y='Predicted_QTY',
                                title=L["demand_chart_title"]),
                        use_container_width=True
                    )

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    L["btn_download"],
                    data      = csv,
                    file_name = f"ROP_{product_code}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime      = 'text/csv',
                )

            except FileNotFoundError as e:
                st.error(L["err_model_not_found"].format(e=e))
                st.info(L["info_run_deploy"])
            except Exception as e:
                st.error(L["err_sim_failed"].format(e=e))
                st.exception(e)

# Footer
st.markdown(
    f"""
    <hr style='margin-top: 50px;'>
    <p style='text-align: center; color: gray;'>
        {L["footer"]}
    </p>
    """,
    unsafe_allow_html=True
)