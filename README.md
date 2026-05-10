# 🏠 House Price Prediction — Machine Learning Model

> A complete end-to-end Machine Learning project that predicts house prices based on various features like area, bedrooms, bathrooms, and amenities.

---

## 📌 Project Overview

This project builds a **Regression ML Model** on a Housing dataset sourced from Kaggle. The goal is to predict the **price of a house** given its features such as size, location quality, furnishing status, and available amenities.

---

## 📂 Dataset

| Detail | Info |
|---|---|
| **Source** | Kaggle — Housing Price Dataset |
| **Rows** | ~545 houses |
| **Target Column** | `price` |
| **Type** | Regression |

### 📊 Features in the Dataset:

| Column | Type | Description |
|---|---|---|
| `price` | Numerical | Price of the house (target variable) |
| `area` | Numerical | Area of the house in sq ft |
| `bedrooms` | Numerical | Number of bedrooms |
| `bathrooms` | Numerical | Number of bathrooms |
| `stories` | Numerical | Number of floors |
| `parking` | Numerical | Number of parking spaces |
| `mainroad` | Binary | Is the house on main road? (yes/no) |
| `guestroom` | Binary | Does it have a guest room? (yes/no) |
| `basement` | Binary | Does it have a basement? (yes/no) |
| `hotwaterheating` | Binary | Hot water heating available? (yes/no) |
| `airconditioning` | Binary | Air conditioning available? (yes/no) |
| `prefarea` | Binary | Is it in a preferred area? (yes/no) |
| `furnishingstatus` | Categorical | furnished / semi-furnished / unfurnished |

---

## 🗂️ Notebook Structure

### 1. 📦 Importing Libraries
All required libraries are imported — `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

---

### 2. 📂 Loading the Dataset
Dataset is loaded using `pd.read_csv()` and basic info like shape, datatypes, and null values are checked.

---

### 3. 🔍 Exploratory Data Analysis (EDA)
Visual analysis of the dataset using multiple graphs:

- **Price Distribution** — Histogram with KDE to understand price spread and skewness
- **Area vs Price** — Scatter plot to see if bigger area means higher price
- **Bedrooms vs Price** — Box plot to compare price across bedroom counts
- **Bathrooms vs Price** — Box plot to compare price across bathroom counts
- **Furnishing Status vs Price** — Bar plot to see price difference by furnishing
- **Air Conditioning vs Price** — Bar plot to see AC premium on price
- **Mainroad & Prefarea vs Price** — Side by side bar plots for location impact
- **Correlation Heatmap** — To find which features are most correlated with price
- **Pairplot** — All numerical features plotted against price in one shot

---

### 4. 🧹 Data Preprocessing

#### Binary Encoding
All yes/no columns converted to 0 and 1:
```
mainroad, guestroom, basement, hotwaterheating,
airconditioning, prefarea → yes=1, no=0
```

#### One Hot Encoding
`furnishingstatus` has 3 unordered categories so **OneHotEncoder from sklearn** is used:
```
furnished      → [0, 0]
semi-furnished → [1, 0]
unfurnished    → [0, 1]
```

---


### 6. ✂️ Train Test Split
Data split into **80% training** and **20% testing** using `train_test_split`.

---

### 7. 🤖 ML Models Used

Four models trained and compared:

| Model | Type | Notes |
|---|---|---|
| **Ridge Regression** | Linear | L2 regularization |
| **Lasso Regression** | Linear | L1 regularization, drops weak features |
| **Random Forest** | Ensemble | Multiple decision trees |


---

### 8. 🔁 K-Fold Cross Validation
**5-Fold Cross Validation** used for reliable model evaluation — each model is trained and tested on 5 different data splits and average R² is reported.

---

### 9. 🎯 Hyperparameter Tuning

- **GridSearchCV** used for Ridge and Lasso (smaller parameter grids)
- **RandomizedSearchCV** used for Random Forest  (larger parameter grids — faster)

Best parameters found automatically for each model.

---

### 10. 📊 Final Model Evaluation

Best model — **Random Forest** with tuned parameters:

| Metric | Value |
|---|---|
| **R² Score** | 0.60 |
| **RMSE** | ₹1,417,091|

#### Evaluation Includes:
- Actual vs Predicted price table with error %
- Residual Plot — checks if errors are random (good model behavior)
- Actual vs Predicted scatter plot

---

### 11. 🏠 Predicting a New House
Final trained model used to predict price of a **brand new unseen house** by passing feature values manually.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML models, preprocessing, evaluation |

---

## 📈 Results Summary

| Model | Cross Val R² | Final Test R² |
|---|---|---|
| Ridge Regression | ~0.63 | - |
| Lasso Regression | ~0.62 | - |
| Random Forest | **0.6476** | **0.6027** |


> **Note:** The slight drop from Cross Val R² to Final R² is expected — Cross Validation uses the full dataset in folds while Final evaluation uses a truly unseen 20% test set.

---

## 💡 Key Learnings

- Binary columns must be encoded before feeding to ML models
- One Hot Encoding is preferred over Label Encoding for unordered categories
- GridSearchCV is great for small param grids; RandomizedSearchCV saves time for large ones
- K-Fold Cross Validation gives more reliable accuracy than a single train/test split
- R² and RMSE are the right metrics for Regression (not Accuracy)
- Small datasets (~545 rows) naturally cap model performance around 60-65%

---

## 🚀 How to Run

```bash
# 1. Clone or download the notebook
# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Download dataset from Kaggle
[# https://www.kaggle.com/datasets/yasserh/housing-prices-dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data)

# 4. Run the notebook top to bottom
jupyter notebook house_price_prediction.ipynb
```

---

## 👤 Author
Made with sincerity as a beginner ML project on Kaggle Housing Dataset.
