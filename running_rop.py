import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta, date  


# Page configuration
st.set_page_config(page_title="Inventory Management System", layout="wide", initial_sidebar_state="expanded")

# Tentukan direktori dasar relatif terhadap lokasi file ini
BASE = Path(__file__).parent
DATA_DIR = BASE / "Data"
MODELS_DIR = BASE / "saved_models"  # Sesuai nama folder Anda: 'saved_models'

@st.cache_resource
def load_components():
    model = joblib.load(MODELS_DIR / "final_model_XGBoost.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    feature_list = joblib.load(MODELS_DIR / "feature_list.joblib")
    return model, scaler, label_encoder, feature_list

@st.cache_data
def load_data():
    try:
        data_path = DATA_DIR / "rekap_penjualan.xlsx"
        data = pd.read_excel(data_path)
        data['Tanggal'] = pd.to_datetime(data['Tanggal'])
        return data
    except:
        # Sample data for demo
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Tanggal': dates,
            'KODE': np.random.choice(['P001', 'P002', 'P003'], len(dates)),
            'NAMA BARANG': np.random.choice(['Product A', 'Product B', 'Product C'], len(dates)),
            'QTY': np.random.randint(10, 100, len(dates)),
            'KATEGORI': np.random.choice(['Category 1', 'Category 2'], len(dates)),
            'SUPPLIER': np.random.choice(['Supplier A', 'Supplier B'], len(dates))
        })
        return sample_data

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
        'rop': rop,
        'safety_stock': safety_stock,
        'avg_daily_demand': avg_daily_demand,
        'std_daily_demand': std_daily_demand,
        'z_score': z_score,
        'lead_time_demand': avg_daily_demand * lead_time_days
    }
def estimate_monthly_demand(data, product_code, months_back=6):
    """
    Estimasi rata-rata permintaan bulanan berdasarkan data historis.
    """
    df = data[data['KODE'] == product_code].copy()
    df = df.sort_values('Tanggal')
    
    if df.empty:
        return 0
    
    end_date = df['Tanggal'].max()
    start_date = end_date - pd.DateOffset(months=months_back)
    df_recent = df[df['Tanggal'] >= start_date]
    
    if df_recent.empty:
        df_recent = df  # fallback ke seluruh data
    
    # Total demand dalam periode (dalam bulan desimal)
    total_days = (df_recent['Tanggal'].max() - df_recent['Tanggal'].min()).days
    if total_days == 0:
        total_days = 30
    total_months = max(total_days / 30.0, 1.0)
    
    total_qty = df_recent['QTY'].sum()
    monthly_demand = total_qty / total_months
    return max(1, round(monthly_demand))
# ========================================
# Fungsi Prediksi (sesuai kode asli Anda)
# ========================================
def predict_demand_rop(
    data, product_code, product_name, start_date, period_days,
    lead_time, initial_stock, model, scaler, feature_list, eoq
):
    # period_days: total days to simulate (30, 90, 180)
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    elif isinstance(start_date, date):
        start_date = pd.Timestamp(start_date)
    elif isinstance(start_date, datetime):
        start_date = pd.Timestamp(start_date)
    else:
        start_date = pd.Timestamp(start_date)

    prediction_dates = [start_date + timedelta(days=i) for i in range(period_days)]

    data['Tanggal'] = pd.to_datetime(data['Tanggal'])
    df_code = data[data['KODE'] == product_code].copy()
    df_code = df_code.sort_values(by='Tanggal')

    df_code['Lag_1_days'] = df_code['QTY'].shift(1)
    df_code['Lag_2_days'] = df_code['QTY'].shift(2)
    df_code['Rolling_Mean_3'] = df_code['QTY'].rolling(window=3).mean()
    df_code = df_code.dropna(subset=['Lag_1_days', 'Lag_2_days', 'Rolling_Mean_3'])

    all_features = feature_list['all_features']
    top_features = feature_list['top_features']

    results = []
    remaining_stock = initial_stock
    order_schedule = {}
    order_frequency = 0

    # Hitung std dev dari 6 bulan terakhir sebelum start_date
    six_months_back = start_date - pd.DateOffset(months=6)
    df_6m = df_code[(df_code['Tanggal'] >= six_months_back) & 
                    (df_code['Tanggal'] < start_date)]
    std_dev = df_6m['QTY'].std()
    if pd.isna(std_dev) or std_dev == 0:
        std_dev = df_code['QTY'].std() or 1.0

    for date_pred in prediction_dates:
        date_pred = pd.Timestamp(date_pred)

        # Terima barang yang datang hari ini
        if date_pred in order_schedule:
            remaining_stock += order_schedule[date_pred]
            del order_schedule[date_pred]

        df_prev = df_code[df_code['Tanggal'] < date_pred].copy()
        if len(df_prev) < 3:
            results.append({
                'Date': date_pred.date(),
                'Product_Name': product_name,
                'Predicted_QTY': 0,
                'ROP': 0,
                'Safety_Stock': 0,
                'Remaining_Stock': remaining_stock,
                'Status': 'Not enough historical data',
                'Order_Note': ''
            })
            continue

        lag_1 = df_prev.iloc[-1]['QTY']
        lag_2 = df_prev.iloc[-2]['QTY']
        rolling_mean_3 = df_prev.iloc[-3:]['QTY'].mean()

        features_pred = pd.DataFrame([{
            'Rolling_Mean_3': rolling_mean_3,
            'Lag_2_days': lag_2,
            'Lag_1_days': lag_1,
            'KODE': product_code,
            'KATEGORI': 0,
            'SUPPLIER': 0,
            'Month': date_pred.month,
            'Day': date_pred.day
        }])

        features_scaled = scaler.transform(features_pred[all_features])
        features_top = pd.DataFrame(features_scaled, columns=all_features)[top_features]
        pred_qty = model.predict(features_top)[0]
        pred_qty = max(0, pred_qty)

        # Gunakan fungsi ROP Anda
        daily_history = df_prev['QTY'].tail(180).values  # ambil max 6 bulan
        if len(daily_history) == 0:
            daily_history = [pred_qty]
        rop_info = calculate_rop_ss(daily_history, lead_time, service_level=0.95)
        rop = rop_info['rop']
        safety_stock = rop_info['safety_stock']

        remaining_stock = max(0, remaining_stock - pred_qty)

        status = "Sufficient"
        order_note = ""
        pending_order = any(t > date_pred for t in order_schedule.keys())

        if remaining_stock <= rop and not pending_order:
            order_qty = eoq
            arrival_date = date_pred + pd.Timedelta(days=lead_time)
            order_schedule[arrival_date] = order_qty
            order_note = f"Order {order_qty} units (arrives {arrival_date.date()})"
            order_frequency += 1
            status = "Reorder Required"
        elif pending_order:
            status = "Awaiting Order Arrival"
            order_note = "Order in transit"

        results.append({
            'Date': date_pred.date(),
            'Product_Name': product_name,
            'Predicted_QTY': round(pred_qty),
            'ROP': round(rop),
            'Safety_Stock': round(safety_stock),
            'Remaining_Stock': round(remaining_stock),
            'Status': status,
            'Order_Note': order_note
        })

        df_code = pd.concat([df_code, pd.DataFrame([{
            'Tanggal': date_pred,
            'QTY': pred_qty,
            'KODE': product_code,
            'NAMA BARANG': product_name
        }])], ignore_index=True)

    result_df = pd.DataFrame(results)
    result_df.attrs['order_frequency'] = order_frequency
    return result_df, rop_info

# Load data
data = load_data()
code_list = data['KODE'].unique().tolist()

# ========================================
# Navigation
# ========================================
st.set_page_config(page_title="Demand & Inventory Dashboard", layout="wide")
page = st.sidebar.radio("Navigation", ["📊 Overview", "📦 Inventory Simulation"])

# ========================================
# PAGE 1: Overview with Dynamic Date Filter
# ========================================
if page.startswith("📊"):
    st.title("📊 Sales Overview & Business Insights")

    min_date = data['Tanggal'].min().date()
    max_date = data['Tanggal'].max().date()
    col1, col2 = st.columns(2)
    with col1:
        start_filter = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_filter = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    if start_filter > end_filter:
        st.error("Start date must be before end date.")
        st.stop()

    filtered_data = data[(data['Tanggal'].dt.date >= start_filter) & (data['Tanggal'].dt.date <= end_filter)].copy()
    date_range_days = (end_filter - start_filter).days

    # --- KPIs ---
    total_sales = filtered_data['QTY'].sum()
    unique_products = filtered_data['KODE'].nunique()

    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.metric("Total Sales", f"{total_sales:,.0f}")
    col_k2.metric("Products", unique_products)
    col_k3.metric("Period", f"{start_filter} → {end_filter}")

    # --- Best & Worst Products ---
    if not filtered_data.empty:
        prod_agg = filtered_data.groupby('KODE').agg(
            Total_Qty=('QTY', 'sum'),
            Product_Name=('NAMA BARANG', 'first')
        ).reset_index().sort_values('Total_Qty', ascending=False)

        st.subheader("🏆 Top & Bottom Products")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Top 5**")
            st.dataframe(prod_agg.head(5)[['Product_Name', 'Total_Qty']], use_container_width=True)
        with col_b:
            st.write("**Bottom 5**")
            st.dataframe(prod_agg.tail(5)[['Product_Name', 'Total_Qty']], use_container_width=True)

        # --- 🔽 TAMBAHAN: Filter by Product or Category (for Sales Trend) ---
        st.subheader("🔍 Filter Sales Trend")
        filter_type = st.radio(
            "Filter by:",
            options=["None", "Product", "Category"],
            index=0,
            horizontal=True
        )

        visual_data = filtered_data.copy()

        if filter_type == "Product":
            selected_code = st.selectbox("Select Product", sorted(filtered_data['KODE'].unique()))
            visual_data = filtered_data[filtered_data['KODE'] == selected_code]
            
            # Ambil nama produk (asumsi: nama konsisten per KODE)
            product_name = visual_data['NAMA BARANG'].iloc[0] if not visual_data.empty else "Unknown"
            st.markdown(f"**Selected Product:** `{selected_code}` – **{product_name}**")

        elif filter_type == "Category":
            if 'KATEGORI' in filtered_data.columns:
                selected_cat = st.selectbox("Select Category", sorted(filtered_data['KATEGORI'].dropna().unique()))
                visual_data = filtered_data[filtered_data['KATEGORI'] == selected_cat]
                st.markdown(f"**Selected Category:** **{selected_cat}**")
            else:
                st.warning("Column 'KATEGORI' not found in dataset.")
                visual_data = filtered_data

        # --- 🔽 Diperbarui: Sales Trend dengan Pilihan Daily/Monthly ---
        st.subheader("📈 Sales Trend")

        # Pilihan tampilan: Daily atau Monthly
        view_option = st.radio(
            "View as:",
            options=["Daily", "Monthly"],
            index=0,  # default: Daily
            horizontal=True
        )

        if view_option == "Daily":
            # Tampilkan data per hari (dari data yang sudah difilter)
            daily_data = visual_data.groupby('Tanggal')['QTY'].sum().reset_index()
            if not daily_data.empty:
                fig = px.line(daily_data, x='Tanggal', y='QTY', title='Daily Sales')
            else:
                fig = go.Figure().update_layout(title="No daily sales data to display")
        else:  # Monthly
            # Agregasi ke level bulan
            temp_data = visual_data.copy()
            temp_data['MonthYear'] = temp_data['Tanggal'].dt.to_period('M')
            monthly_data = temp_data.groupby('MonthYear')['QTY'].sum().reset_index()
            monthly_data['MonthYear'] = monthly_data['MonthYear'].astype(str)
            monthly_data['MonthYear'] = pd.to_datetime(monthly_data['MonthYear'])
            if not monthly_data.empty:
                fig = px.line(monthly_data, x='MonthYear', y='QTY', title='Monthly Sales')
                fig.update_xaxes(tickformat="%b %Y", dtick="M1")
            else:
                fig = go.Figure().update_layout(title="No monthly sales data to display")

        st.plotly_chart(fig, use_container_width=True)

        # --- Histogram ---
        st.subheader("📊 Demand Distribution")
        st.plotly_chart(px.histogram(filtered_data, x='QTY', nbins=50), use_container_width=True)

        # --- 🔽 TAMBAHAN: Top 10 Products ---
        st.subheader("🥇 Top 10 Best-Selling Products")
        top10 = prod_agg.head(10)
        if not top10.empty:
            fig_top10 = px.bar(
                top10,
                x='Total_Qty',
                y='Product_Name',
                orientation='h',
                title="Top 10 Products by Sales Volume"
            )
            fig_top10.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_top10, use_container_width=True)

        # --- 🔽 TAMBAHAN: Pie Chart by Category ---
        if 'KATEGORI' in filtered_data.columns:
            st.subheader("🍰 Sales by Product Category")
            cat_sales = filtered_data.groupby('KATEGORI')['QTY'].sum().reset_index()
            if not cat_sales.empty:
                fig_pie = px.pie(
                    cat_sales,
                    values='QTY',
                    names='KATEGORI',
                    title="Sales Distribution by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

# ========================================
# PAGE 2: Inventory Simulation
# ========================================
elif page.startswith("📦"):
    st.title("📦 Inventory Simulation: EOQ & ROP")

    product_code = st.selectbox("Select Product Code", code_list)
    product_name = data.loc[data['KODE'] == product_code, 'NAMA BARANG'].values[0]
    st.markdown(f"**Product:** {product_name}")

    tab_eoq, tab_rop = st.tabs(["🔁 EOQ Calculator", "⚠️ ROP Simulation"])

    # ---------------- EOQ Tab ----------------
    with tab_eoq:
        st.subheader("Economic Order Quantity (EOQ) – Auto Estimated")

        # Estimasi demand bulanan dari data historis
        monthly_demand_est = estimate_monthly_demand(data, product_code, months_back=6)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Estimated Monthly Demand:** {monthly_demand_est:,} units (from last 6 months)")
        with col_b:
            st.write("")

        ordering_cost = st.number_input("Ordering Cost per Order (IDR)", min_value=1, value=50000)
        holding_cost = st.number_input("Holding Cost per Unit per Month (IDR)", min_value=1, value=200)

        eoq = calculate_eoq(monthly_demand_est, ordering_cost, holding_cost)
        if eoq == 0:
            st.warning("Unable to calculate EOQ (check input values).")
        else:
            st.success(f"✅ **Recommended EOQ:** {eoq:,} units")
        
        # Simpan ke session state untuk digunakan di tab ROP
        st.session_state['eoq_value'] = eoq
        st.session_state['monthly_demand'] = monthly_demand_est

    # ---------------- ROP Tab ----------------
    with tab_rop:
        st.subheader("ROP Simulation with EOQ Restocking")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.today().date())
        with col2:
            use_custom = st.checkbox("Custom end date")
            if use_custom:
                end_date = st.date_input("End Date", start_date + timedelta(days=30))
                if end_date <= start_date:
                    st.error("End date must be after start date.")
                    period_days = 1
                else:
                    period_days = (end_date - start_date).days + 1  # inklusif
            else:
                period_option = st.selectbox("Simulation Period", ["1 Month", "3 Months", "6 Months"])
                period_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180}
                period_days = period_map[period_option]

        col3, col4 = st.columns(2)
        with col3:
            lead_time = st.number_input("Lead Time (days)", min_value=1, value=3)
        with col4:
            initial_stock = st.number_input("Initial Stock", min_value=0, value=100)

        eoq_value = st.session_state.get('eoq_value', 100)
        st.info(f"Restocking quantity: **{eoq_value} units (EOQ)**")

        if st.button("🚀 Run Simulation"):
            try:
                model, scaler, label_encoder, feature_list = load_components()
                result_df, rop_info = predict_demand_rop(
                    data=data,
                    product_code=product_code,
                    product_name=product_name,
                    start_date=start_date,
                    period_days=period_days,
                    lead_time=lead_time,
                    initial_stock=initial_stock,
                    model=model,
                    scaler=scaler,
                    feature_list=feature_list,
                    eoq=eoq_value
                )

                # --- Parameters ---
                st.subheader("⚙️ Key Parameters")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Avg Daily Demand", f"{rop_info['avg_daily_demand']:.1f}")
                p2.metric("Std Dev", f"{rop_info['std_daily_demand']:.2f}")
                p3.metric("Service Level", "95%")
                p4.metric("Z-Score", f"{rop_info['z_score']:.2f}")

                # --- Summary ---
                st.subheader("📋 Simulation Summary")
                total_demand = result_df['Predicted_QTY'].sum()
                avg_demand = result_df['Predicted_QTY'].mean()
                orders_placed = result_df.attrs['order_frequency']

                s1, s2, s3 = st.columns(3)
                s1.metric("Total Forecast Demand", f"{total_demand:.0f}")
                s2.metric("Avg Daily Demand", f"{avg_demand:.1f}")
                s3.metric("Orders Placed", orders_placed)

                # --- Results Table ---
                st.subheader("📅 Simulation Results")
                st.dataframe(result_df, use_container_width=True)

                # --- Grafik utama: Predicted_QTY, ROP, Remaining_Stock ---
                st.subheader("📉 Inventory Simulation Over Time")
                if not result_df.empty:
                    chart_data = result_df.set_index('Date')[['Predicted_QTY', 'ROP', 'Remaining_Stock']]
                    st.line_chart(chart_data)
                else:
                    st.info("No simulation data to display.")

                # --- Grafik prediksi terpisah (tetap dipertahankan) ---
                st.subheader("📈 Predicted Demand Only")
                if not result_df.empty:
                    fig_pred = px.line(result_df, x='Date', y='Predicted_QTY', title='Predicted Daily Demand')
                    st.plotly_chart(fig_pred, use_container_width=True)

                # --- Download ---
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results",
                    data=csv,
                    file_name=f"ROP_{product_code}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )

            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                st.exception(e)

# Footer
st.markdown(
    """
    <hr style='margin-top: 50px;'>
    <p style='text-align: center; color: gray;'>
        © 2025 Andre Nugraha. All rights reserved.
    </p>
    """,
    unsafe_allow_html=True
)