# Inventory and Product Grouping System

## 📌 Project Overview

The **Inventory and Product Grouping System** is a machine learning project that uses K-Means clustering to group similar products based on inventory levels, total units sold, price, and demand. It helps businesses understand product groups and improve inventory management and stock planning.

## 🎯 Business Problem

Businesses often struggle to identify products with different sales and demand patterns. This system groups similar products to help businesses make better inventory planning and stock management decisions.

## 🛠️ Technologies Used

- **Python** – Core programming language
- **Pandas & NumPy** – Data processing and numerical operations
- **Scikit-learn** – K-Means clustering and feature scaling
- **Joblib** – Saving and loading trained models
- **Flask** – Web application development
- **HTML & CSS** – User interface design
- **Jupyter Notebook** – Model experimentation

## ⚙️ Machine Learning Workflow

1. Load and preprocess the sales dataset.
2. Aggregate records by Product ID.
3. Select inventory level, units sold, price, and demand as features.
4. Scale numerical features using StandardScaler.
5. Apply K-Means clustering with three clusters.
6. Save the trained model and scaler.
7. Predict product groups through a Flask web application.

## 📂 Project Structure

```text
Inventory_Product_Grouping_System/
├── data/
│   └── sales_data.csv
├── notebooks/
│   └── inventory_experiment.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── prediction.py
├── models/
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Inventory_Product_Grouping_System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the clustering model

```bash
python -m src.clustering
```

This generates the trained K-Means model, scaler, and feature configuration in the `models/` directory.

### 4. Run the Flask application

```bash
python app.py
```

## 📊 Expected Outcome

The system assigns products to groups such as:

- High-Demand, High-Sales Products
- Lower-Demand, Higher-Price Products
- Moderate-Demand Products

These labels describe the clusters observed during experimentation and should be verified against the trained model's cluster statistics.

## 💡 Future Improvements

- Add interactive inventory reports.
- Improve cluster interpretation using additional business rules.
- Recommend stock levels based on demand patterns.
- Add visualizations for product groups.

## 👨‍💻 Author

Developed as a machine learning project focused on product clustering and inventory management.
