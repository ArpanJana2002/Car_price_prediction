## 🚗 Old Car Price Prediction using Machine Learning

This project uses machine learning techniques to predict the **resale price of used cars** based on various features like brand, model, fuel type, transmission, and more.

The dataset used is a cleaned CSV of car listings, and the model is trained using **scikit-learn**. The project includes preprocessing, model training, evaluation, and prediction on new data.

---

### 📁 Files

* `car-price-prediction.ipynb` – Main Jupyter Notebook with full code pipeline
* `Car Price.csv` – Dataset (you may upload this separately if needed)
---

### 🔍 Features Used

* `Brand`
* `Model`
* `Year`
* `KM_Driven`
* `Fuel`
* `Seller_Type`
* `Transmission`
* `Owner`

---

### 🛠️ Tools & Libraries

* Python 3
* pandas
* scikit-learn
* matplotlib , seaborn
* Jupyter Notebook 

---

### 🔄 Workflow Summary

1. **Data Loading & Exploration**
2. **Data Cleaning & Encoding**
3. **Feature Scaling (StandardScaler)**
4. **Train/Test Split**
5. **Modeling:**

   * Linear Regression
   * Random Forest Regressor
6. **Model Evaluation:**

   * R² Score
   * Mean Squared Error (MSE)
   * Mean Absolute Error (MAE)
7. **Prediction on New Input**

---

### 📈 Sample Prediction

```python
new_data = pd.DataFrame([[
    "Maruti", "Maruti 800 AC", 2007, 70000,
    "Petrol", "Individual", "Manual", "First Owner"
]], columns=x_train.columns)

predicted_price = model.predict(new_data)
print(f"Predicted Selling Price: {predicted_price[0]}")
```

---

### 🚀 Getting Started

To run the notebook:

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/old-car-price-prediction.git
   cd old-car-price-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:

   ```bash
   jupyter notebook car-price-prediction.ipynb
   ```

---

### 🧠 Model Accuracy (Example)

* Linear Regression R² Score: \~85% on training set
* Random Forest R² Score: \~92% on test set

---
