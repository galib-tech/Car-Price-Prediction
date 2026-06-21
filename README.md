# 🚗 Car Price Prediction

A machine learning web app that predicts the resale price of used cars based on brand, manufacturing year, kilometers driven, and fuel type. Built with a Random Forest Regressor and deployed using Streamlit.

🔗 **Live Demo:** [car-price-prediction-hyeku4qlksrvgfc4fftmlg.streamlit.app](https://car-price-prediction-hyeku4qlksrvgfc4fftmlg.streamlit.app/)

---

## 📌 Overview

This project predicts used car prices using real-world listing data scraped from Quikr. The raw dataset was heavily unstructured — containing inconsistent formats, missing values, and invalid entries — and required extensive cleaning before modeling.

A **Random Forest Regressor** was trained on the cleaned data, achieving an **R² score of ~0.63**, a significant improvement over the initial Linear Regression baseline.

---

## 🧹 Key Challenges & Solutions

| Challenge | Solution |
|---|---|
| Inconsistent `year`, `Price`, `kms_driven` formats | Regex-based cleaning and type casting |
| High-cardinality categorical features (`name`, `company`) | Target Encoding with smoothing to prevent overfitting |
| Rare category (`LPG` fuel type) causing encoder errors | `handle_unknown="ignore"` in OneHotEncoder |
| Unstable evaluation scores across runs | Fixed `random_state` + cross-validation for reliable reporting |
| Non-linear price depreciation pattern | Switched from Linear Regression to Random Forest |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Modeling:** Scikit-learn (RandomForestRegressor), Category Encoders (TargetEncoder)
- **Deployment:** Streamlit
- **Serialization:** Pickle

---

## 📂 Project Structure

```
Car-Price-Prediction/
├── app.py              # Streamlit application
├── clean_car.csv        # Cleaned dataset used for training
├── pipe.pkl              # Trained model pipeline (encoding + Random Forest)
├── requirements.txt      # Project dependencies
└── README.md
```

---

## ⚙️ How It Works

1. **Data Cleaning** — Raw scraped data is filtered for valid years, numeric prices, and proper kilometer readings.
2. **Feature Encoding** — Categorical features (`company`, `fuel_type`) are encoded using Target Encoding and One-Hot Encoding respectively.
3. **Model Training** — A Random Forest Regressor is trained on the processed features to capture non-linear pricing patterns.
4. **Deployment** — The trained pipeline is serialized with Pickle and served through an interactive Streamlit interface.

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/galib-tech/Car-Price-Prediction.git
cd Car-Price-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| R² Score | ~0.63 |

> Evaluated using cross-validation across multiple train-test splits for a reliable, non-cherry-picked estimate.

---

## 🔮 Future Improvements

- Incorporate additional features (engine capacity, transmission type, owner count)
- Expand dataset size for better generalization
- Experiment with ensemble methods (Voting Regressor, XGBoost)

---

## 👤 Author

**Asadullah Al Galib**                                                                          
GitHub: [@galib-tech](https://github.com/galib-tech)
