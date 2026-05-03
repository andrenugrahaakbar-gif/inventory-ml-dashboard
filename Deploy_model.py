# ======================
# 1. IMPORT LIBRARIES
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_percentage_error,
                              mean_squared_error, r2_score,
                              mean_absolute_error)
from sklearn.model_selection import learning_curve
import joblib
import os

# ======================
# 2. LOAD DATA
# ======================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(BASE_DIR, "Data", "rekap_penjualan.xlsx")

product_data = pd.read_excel(file_path)
product_data['Tanggal'] = pd.to_datetime(product_data['Tanggal'])

print(f"Data loaded: {len(product_data):,} rows, {product_data['KODE'].nunique()} SKUs")
print(f"Date range : {product_data['Tanggal'].min().date()} → {product_data['Tanggal'].max().date()}")

# ======================
# 3. LABEL ENCODING
# ======================
\
encoders = {}

for col in ['KODE', 'KATEGORI', 'SUPPLIER']:
    enc = LabelEncoder()
    product_data[col] = enc.fit_transform(product_data[col].astype(str))
    encoders[col] = enc
    print(f"  Encoded '{col}': {len(enc.classes_)} unique values")

# ======================
# 4. OUTLIER HANDLING
# ======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=product_data['QTY'])
plt.title('QTY Before Outlier Handling')

def cap_outliers_adaptive(group):
    """
    IQR * 1.5 untuk SKU stabil (CV ≤ 0.5)
    IQR * 2.5 untuk SKU volatile  (CV > 0.5)
    Mencegah pemotongan sinyal peak demand yang legitimate.
    """
    Q1  = group['QTY'].quantile(0.25)
    Q3  = group['QTY'].quantile(0.75)
    IQR = Q3 - Q1
    mean_ = group['QTY'].mean()
    std_  = group['QTY'].std()
    cv    = std_ / mean_ if mean_ > 0 else 0

    multiplier  = 2.5 if cv > 0.5 else 1.5
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    group = group.copy()
    group['QTY'] = np.clip(group['QTY'], lower_bound, upper_bound)
    return group

product_data = (
    product_data
    .groupby('KODE', group_keys=False)
    .apply(cap_outliers_adaptive)
)

plt.subplot(1, 2, 2)
sns.boxplot(x=product_data['QTY'])
plt.title('QTY After Adaptive Outlier Handling')
plt.tight_layout()
plt.show()

# ======================
# 5. FEATURE ENGINEERING
# ======================
product_data = product_data.sort_values(['KODE', 'Tanggal']).reset_index(drop=True)

# Temporal
product_data['Day']       = product_data['Tanggal'].dt.day
product_data['DayOfWeek'] = product_data['Tanggal'].dt.dayofweek
product_data['Month']     = product_data['Tanggal'].dt.month
product_data['Year']      = product_data['Tanggal'].dt.year

# Lag features (per SKU)
for lag in [1, 2, 7]:
    product_data[f'Lag_{lag}_days'] = (
        product_data.groupby('KODE')['QTY'].shift(lag)
    )

# Rolling features (per SKU)
for window in [3, 7]:
    product_data[f'Rolling_Mean_{window}'] = (
        product_data.groupby('KODE')['QTY']
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    product_data[f'Rolling_Std_{window}'] = (
        product_data.groupby('KODE')['QTY']
        .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    )

before = len(product_data)
product_data = product_data.dropna().reset_index(drop=True)
print(f"\nDropped {before - len(product_data)} rows (NaN from lag). Remaining: {len(product_data):,}")

# ======================
# 6. FEATURE SETS & TRAIN-TEST SPLIT
# ======================
NUMERIC_FEATURES     = ['Lag_1_days', 'Lag_2_days', 'Lag_7_days',
                         'Rolling_Mean_3', 'Rolling_Mean_7',
                         'Rolling_Std_3', 'Rolling_Std_7']
CATEGORICAL_FEATURES = ['KODE', 'KATEGORI', 'SUPPLIER']
TEMPORAL_FEATURES    = ['Month', 'Day', 'DayOfWeek']

ALL_FEATURES = CATEGORICAL_FEATURES + TEMPORAL_FEATURES + NUMERIC_FEATURES

# Time-based split (kronologis)
product_data_sorted = product_data.sort_values('Tanggal').reset_index(drop=True)
split_date = product_data_sorted['Tanggal'].quantile(0.8)
print(f"\nTrain/test split date: {split_date.date()}")

train_mask = product_data_sorted['Tanggal'] <  split_date
test_mask  = product_data_sorted['Tanggal'] >= split_date

X_all = product_data_sorted[ALL_FEATURES]
y_all = product_data_sorted['QTY']

X_train = X_all[train_mask].reset_index(drop=True)
X_test  = X_all[test_mask].reset_index(drop=True)
y_train = y_all[train_mask].reset_index(drop=True)
y_test  = y_all[test_mask].reset_index(drop=True)

print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

scaler = StandardScaler()

X_train_numeric_arr = X_train[NUMERIC_FEATURES].values
X_test_numeric_arr  = X_test[NUMERIC_FEATURES].values
 
scaler.fit(X_train_numeric_arr) 
 
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
 
X_train_scaled[NUMERIC_FEATURES] = scaler.transform(X_train_numeric_arr)
X_test_scaled[NUMERIC_FEATURES]  = scaler.transform(X_test_numeric_arr)
 
print(f"Scaler fitted. n_features: {scaler.n_features_in_}")
print(f"feature_names_in_ ada: {hasattr(scaler, 'feature_names_in_')}")

# Cross-validation temporal
tscv = TimeSeriesSplit(n_splits=5, gap=1)

# ======================
# 7. EVALUASI BASELINE MODELS
# ======================
def get_baseline_models():
    """Selalu kembalikan instance baru agar state tidak tumpang tindih."""
    return {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'XGBoost'      : XGBRegressor(random_state=42, eval_metric='rmse'),
    }


def evaluate_models(X_tr, y_tr, X_te, y_te, models: dict) -> dict:
    """
    Evaluasi kumpulan model. Metrik: MAPE, MAE, RMSE, R².
    FIX #5: MAE ditambahkan sebagai metrik utama karena lebih robust
    terhadap nilai aktual kecil dibanding MAPE.
    """
    results = {}
    for name, model in models.items():
        print(f"\n  [{name}] fitting...")
        model.fit(X_tr, y_tr)

        tr_pred = model.predict(X_tr)
        te_pred = model.predict(X_te)

        tr_pred = np.maximum(0, tr_pred)
        te_pred = np.maximum(0, te_pred)

        results[name] = {
            'model'     : model,
            'train_mape': mean_absolute_percentage_error(y_tr, tr_pred) * 100,
            'test_mape' : mean_absolute_percentage_error(y_te, te_pred) * 100,
            'train_mae' : mean_absolute_error(y_tr, tr_pred),
            'test_mae'  : mean_absolute_error(y_te, te_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_tr, tr_pred)),
            'test_rmse' : np.sqrt(mean_squared_error(y_te, te_pred)),
            'train_r2'  : r2_score(y_tr, tr_pred),
            'test_r2'   : r2_score(y_te, te_pred),
        }
        r = results[name]
        print(f"         Test → MAPE: {r['test_mape']:.2f}%  "
              f"MAE: {r['test_mae']:.2f}  RMSE: {r['test_rmse']:.2f}  R²: {r['test_r2']:.4f}")
    return results


print("\n===== BASELINE EVALUATION (all features) =====")
baseline_results = evaluate_models(
    X_train_scaled, y_train,
    X_test_scaled,  y_test,
    get_baseline_models()
)

# Visualisasi baseline
metrics_df = pd.DataFrame({
    'Model'     : list(baseline_results.keys()),
    'Train MAPE': [v['train_mape'] for v in baseline_results.values()],
    'Test MAPE' : [v['test_mape']  for v in baseline_results.values()],
    'Train RMSE': [v['train_rmse'] for v in baseline_results.values()],
    'Test RMSE' : [v['test_rmse']  for v in baseline_results.values()],
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
idx = np.arange(len(metrics_df))
bw  = 0.35
for ax, col_tr, col_te, ylabel in zip(
    axes,
    ['Train MAPE', 'Train RMSE'],
    ['Test MAPE',  'Test RMSE'],
    ['MAPE (%)',   'RMSE']
):
    ax.bar(idx,      metrics_df[col_tr], bw, label='Train')
    ax.bar(idx + bw, metrics_df[col_te], bw, label='Test')
    ax.set_xticks(idx + bw / 2)
    ax.set_xticklabels(metrics_df['Model'])
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} — Baseline')
    ax.legend()
plt.tight_layout()
plt.show()

# Best baseline
best_baseline_name, best_baseline_info = min(
    baseline_results.items(), key=lambda x: x[1]['test_mape']
)
print(f"\n✔ Best baseline: {best_baseline_name}  "
      f"(Test MAPE: {best_baseline_info['test_mape']:.2f}%)")

# ======================
# 8. FEATURE IMPORTANCE & TOP FEATURE SELECTION
# ======================
best_baseline_model = best_baseline_info['model']

if hasattr(best_baseline_model, 'feature_importances_'):
    feat_imp = (
        pd.DataFrame({
            'Feature'   : ALL_FEATURES,
            'Importance': best_baseline_model.feature_importances_,
        })
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title(f'Feature Importance — {best_baseline_name}')
    plt.tight_layout()
    plt.show()

    # Ambil top-6, pastikan KODE selalu masuk
    top_features = feat_imp['Feature'].head(5).tolist()
    if 'KODE' not in top_features:
        top_features[-1] = 'KODE'
        print(f"  KODE tidak di top-5, diganti masuk: {top_features}")
    else:
        print(f"  Top features: {top_features}")

    X_train_top = X_train_scaled[top_features]
    X_test_top  = X_test_scaled[top_features]

    # ======================
    # 9. MODELING DENGAN TOP FEATURES
    # ======================
    print("\n===== TOP FEATURE EVALUATION =====")
    top_feature_results = evaluate_models(
        X_train_top, y_train,
        X_test_top,  y_test,
        get_baseline_models()  
    )

    best_top_name, best_top_info = min(
        top_feature_results.items(), key=lambda x: x[1]['test_mape']
    )
    print(f"\n✔ Best with top features: {best_top_name}  "
          f"(Test MAPE: {best_top_info['test_mape']:.2f}%)")

    # ======================
    # 10. HYPERPARAMETER TUNING
    # ======================
    param_grids = {
        'Decision Tree': {
            'max_depth'        : [3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf' : [1, 2, 4, 8],
        },
        'Random Forest': {
            'n_estimators'    : [50, 100, 200, 300],
            'max_depth'       : [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf' : [1, 2, 4],
        },
        'XGBoost': {
            'n_estimators'    : [100, 300, 500],
            'max_depth'       : [4, 6, 8],
            'learning_rate'   : [0.01, 0.05, 0.1, 0.2],
            'subsample'       : [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        },
    }

    def make_model_for_tuning(name):
        if name == 'Decision Tree':
            return DecisionTreeRegressor(random_state=42)
        elif name == 'Random Forest':
            return RandomForestRegressor(random_state=42, n_jobs=-1)
        else:
            return XGBRegressor(random_state=42, eval_metric='rmse', n_jobs=-1)

    print(f"\nTuning {best_top_name}...")
    grid_search = GridSearchCV(
        estimator  = make_model_for_tuning(best_top_name),
        param_grid = param_grids[best_top_name],
        scoring    = 'neg_root_mean_squared_error',
        cv         = tscv,
        n_jobs     = -1,
        verbose    = 1,
    )
    grid_search.fit(X_train_top, y_train)

    final_model   = grid_search.best_estimator_
    best_params   = grid_search.best_params_

    y_tr_pred = np.maximum(0, final_model.predict(X_train_top))
    y_te_pred = np.maximum(0, final_model.predict(X_test_top))

    tuned_results = {
        'train_mape': mean_absolute_percentage_error(y_train, y_tr_pred) * 100,
        'test_mape' : mean_absolute_percentage_error(y_test,  y_te_pred) * 100,
        'train_mae' : mean_absolute_error(y_train, y_tr_pred),
        'test_mae'  : mean_absolute_error(y_test,  y_te_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_tr_pred)),
        'test_rmse' : np.sqrt(mean_squared_error(y_test,  y_te_pred)),
        'train_r2'  : r2_score(y_train, y_tr_pred),
        'test_r2'   : r2_score(y_test,  y_te_pred),
    }

    print(f"\n===== TUNED MODEL =====")
    print(f"Best params : {best_params}")
    print(f"Test MAPE   : {tuned_results['test_mape']:.2f}%")
    print(f"Test MAE    : {tuned_results['test_mae']:.2f}")
    print(f"Test RMSE   : {tuned_results['test_rmse']:.2f}")
    print(f"Test R²     : {tuned_results['test_r2']:.4f}")

    # ======================
    # 11. LEARNING CURVE
    # ======================
    def plot_learning_curve(estimator, X, y, title):
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y,
            cv      = tscv,
            scoring = 'neg_mean_absolute_error',
            n_jobs  = -1,
        )
        plt.figure(figsize=(9, 5))
        plt.plot(train_sizes, -train_scores.mean(axis=1), 'o-', label='Train MAE')
        plt.plot(train_sizes, -val_scores.mean(axis=1),   'o-', label='Val MAE')
        plt.title(title)
        plt.xlabel('Training Set Size')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    plot_learning_curve(
        final_model, X_train_top, y_train,
        f'Learning Curve: {best_top_name} (Tuned)'
    )

    # ======================
    # 12. SUMMARY
    # ======================
    summary_df = pd.DataFrame({
        'Metric'  : ['MAPE (%)', 'MAE', 'RMSE', 'R²'],
        'Training': [tuned_results['train_mape'], tuned_results['train_mae'],
                     tuned_results['train_rmse'], tuned_results['train_r2']],
        'Testing' : [tuned_results['test_mape'],  tuned_results['test_mae'],
                     tuned_results['test_rmse'],  tuned_results['test_r2']],
    })
    print("\n===== FINAL MODEL SUMMARY =====")
    print(f"Model       : {best_top_name} (Tuned)")
    print(f"Top Features: {top_features}")
    print(summary_df.to_string(index=False))

    # ======================
    # 13. SIMPAN MODEL & KOMPONEN
    # ======================
    model_dir = os.path.join(BASE_DIR, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)

    # Model
    joblib.dump(
        final_model,
        os.path.join(model_dir, f'final_model_{best_top_name.replace(" ", "")}.joblib')
    )

    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))

    joblib.dump(encoders, os.path.join(model_dir, 'label_encoders.joblib'))
    joblib.dump(encoders['KODE'], os.path.join(model_dir, 'label_encoder.joblib'))

    feature_list = {
        'all_features'           : ALL_FEATURES,
        'top_features'           : top_features,
        'numeric_features'       : NUMERIC_FEATURES,          
        'numeric_features_order' : NUMERIC_FEATURES,          
        'categorical_features'   : CATEGORICAL_FEATURES,
        'temporal_features'      : TEMPORAL_FEATURES,
        'n_numeric_features'     : len(NUMERIC_FEATURES),     
    }
    print(f"feature_list keys: {list(feature_list.keys())}")
    print(f"numeric_features : {feature_list['numeric_features']}")
    joblib.dump(feature_list, os.path.join(model_dir, 'feature_list.joblib'))

    # Split info
    split_info = {
        'split_date': split_date,
        'train_size': len(X_train),
        'test_size' : len(X_test),
    }
    joblib.dump(split_info, os.path.join(model_dir, 'split_info.joblib'))

    print(f"\n✅ Model & komponen tersimpan di: {model_dir}/")
    for f in os.listdir(model_dir):
        print(f"   {f}")

    # ======================
    # CATATAN UNTUK running_rop.py
    # ======================
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH WAJIB DI running_rop.py (predict_demand_rop):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Load label_encoders (dict) saat load_components():
      label_encoders = joblib.load(MODELS_DIR / "label_encoders.joblib")

2. Encode KODE saat membuat features_pred:
      kode_encoded = label_encoders['KODE'].transform([product_code])[0]

3. Tambahkan DayOfWeek ke features_pred:
      'DayOfWeek': date_pred.dayofweek,

4. Saat scaler.transform(), hanya transform NUMERIC_FEATURES,
   bukan seluruh feature array:
      num_cols = feature_list['numeric_features']
      features_pred[num_cols] = scaler.transform(features_pred[num_cols])

Contoh lengkap payload features_pred yang benar:
      features_pred = pd.DataFrame([{
          'KODE'           : kode_encoded,          # sudah di-encode
          'KATEGORI'       : kat_encoded,            # sudah di-encode
          'SUPPLIER'       : sup_encoded,            # sudah di-encode
          'Month'          : date_pred.month,
          'Day'            : date_pred.day,
          'DayOfWeek'      : date_pred.dayofweek,   # ← TAMBAHKAN INI
          'Lag_1_days'     : lag_1,
          'Lag_2_days'     : lag_2,
          'Lag_7_days'     : lag_7,                  # ← TAMBAHKAN INI
          'Rolling_Mean_3' : rolling_mean_3,
          'Rolling_Mean_7' : rolling_mean_7,
          'Rolling_Std_3'  : rolling_std_3,
          'Rolling_Std_7'  : rolling_std_7,
      }])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

else:
    print("Model tidak memiliki feature_importances_. Evaluasi manual diperlukan.")