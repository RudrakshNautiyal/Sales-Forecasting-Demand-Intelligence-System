# 📈 Sales Forecasting & Demand Intelligence System

A machine learning-powered sales analytics platform that forecasts 3-month demand, detects sales anomalies, and segments products into demand clusters — delivered through a 4-page interactive Streamlit dashboard and an executive-grade business report.

**Internship Project (Weeks 3 & 4) — July 2026**

---

## 🔍 What It Does

| Module | Description |
|---|---|
| **Sales Overview** | Interactive dashboard with region & category filters across 4 years of data |
| **Forecast Explorer** | 3-month sales forecast using Prophet, SARIMA, and XGBoost with confidence intervals |
| **Anomaly Report** | Dual anomaly detection — Isolation Forest + Z-Score — on weekly sales data |
| **Product Demand Segments** | KMeans clustering + PCA to segment 17 sub-categories into demand strategies |

---

## 📊 Key Results

### 3-Month Forecast (Overall Sales)

| Model | Month 1 | Month 2 | Month 3 | Error |
|---|---|---|---|---|
| SARIMA | ~$68,000 | ~$72,000 | ~$95,000 | Lowest MAPE |
| Prophet | ~$66,000 | ~$70,000 | ~$92,000 | Medium MAPE |
| XGBoost | ~$65,000 | ~$69,000 | ~$90,000 | Medium MAPE |

> All three models capture Q4 seasonal spike. SARIMA recommended for production use.

### Top Anomalies Detected

- **Nov 2017** — Sales surged above seasonal range (likely Black Friday / year-end corporate buying)
- **Feb 2016** — Sales dropped below trend (delayed Q1 budget approvals)
- **Sep 2018** — Unusual mid-quarter spike (bulk enterprise orders / promotional campaign)

### Demand Segmentation

| Segment | Sub-Categories | Recommended Action |
|---|---|---|
| High Volume, Stable Demand | Chairs, Storage, Phones | Maintain safety stock, automate reorders |
| Growing Demand | Accessories, Copiers | Increase procurement gradually |
| Low Volume, High Volatility | Fasteners, Labels | Conservative stock, just-in-time ordering |
| Declining Demand | Tables, Bookcases | Reduce stock, run clearance promotions |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Forecasting:** Facebook Prophet, Statsmodels (SARIMAX), XGBoost
- **ML / Analytics:** Scikit-learn (KMeans, IsolationForest, PCA, StandardScaler)
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib

---

## 📂 Project Structure

    Sales-Forecasting-Demand-Intelligence-System/
    ├── app.py                        # 4-page Streamlit dashboard
    ├── requirements.txt
    ├── Datasets/
    │   ├── train_processed.csv       # Cleaned transactional data (2015–2018)
    │   ├── monthly_sales.csv         # Aggregated monthly sales
    │   └── weekly_sales.csv          # Aggregated weekly sales
    ├── notebooks/
    │   └── analysis.ipynb            # Full EDA, model building, evaluation
    ├── summary.pdf                   # Executive business report
    └── README.md

---

## ⚙️ Setup & Run

    git clone https://github.com/RudrakshNautiyal/Sales-Forecasting-Demand-Intelligence-System.git
    cd Sales-Forecasting-Demand-Intelligence-System
    pip install -r requirements.txt
    streamlit run app.py

Visit http://localhost:8501

---

## 📈 Dataset

- **Source:** Superstore Sales Dataset (Kaggle)
- **Period:** 2015–2018 (4 years)
- **Scope:** 4 regions, 3 categories, 17 sub-categories
- **Revenue:** Technology $827K · Furniture $729K · Office Supplies $705K

---

## 💡 Business Insights

- **East region** showed the most consistent year-over-year growth (std = 0.018) — lowest-risk market for expansion
- **November & December** consistently generate peak sales — stock up by mid-October to avoid lost sales
- **Tables & Bookcases** are declining year-over-year — capital tied up here has diminishing return
- Average shipping time is ~4 days across all regions — logistics are well-standardised

---

## ⚠️ Limitations

- Trained on 4 years of historical data — accuracy degrades beyond a 3-month horizon
- Does not account for external shocks (supply chain disruptions, competitor changes, macro events)
- Anomaly detection flags statistical outliers but cannot distinguish genuine demand spikes from data entry errors without manual review

---

## 👤 Author

**Rudraksh Nautiyal**
B.Tech CSE — Jaypee University of Information Technology
[GitHub](https://github.com/RudrakshNautiyal)
