# Mushroom Classification Web App

This project is a **Streamlit web application** that predicts whether a mushroom is **edible or poisonous** using machine learning classifiers. The app allows users to select between Support Vector Machine (SVM), Logistic Regression, and Random Forest classifiers, tune their hyperparameters, and visualize model performance metrics.

## Features

- **Data Preprocessing:** Encodes categorical mushroom data for modeling.
- **Model Selection:** Choose between SVM, Logistic Regression, and Random Forest.
- **Hyperparameter Tuning:** Adjust model parameters from the sidebar.
- **Metrics Visualization:** Plot Confusion Matrix, ROC Curve, and Precision-Recall Curve.
- **Show Raw Data:** Option to display the raw mushroom dataset.

## Getting Started

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- pandas, numpy, scikit-learn

Install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn
```

### Dataset

Place the `mushrooms.csv` dataset in the project root directory.  
You can download it from [Kaggle: Mushroom Classification](https://www.kaggle.com/uciml/mushroom-classification).

### Running the App

```bash
streamlit run app.py
```

Open the provided local URL in your browser to interact with the app.

## Usage

1. Select a classifier from the sidebar.
2. Adjust hyperparameters as desired.
3. Choose which metrics to plot.
4. Click **Classify** to train and evaluate the model.
5. Optionally, check "Show raw data" to view the dataset.

## Project Structure

```
.
├── app.py
├── mushrooms.csv
└── README.md
```

## License

This project is for educational purposes.

---
```# Mushroom Classification Web App

This project is a **Streamlit web application** that predicts whether a mushroom is **edible or poisonous** using machine learning classifiers. The app allows users to select between Support Vector Machine (SVM), Logistic Regression, and Random Forest classifiers, tune their hyperparameters, and visualize model performance metrics.

## Features

- **Data Preprocessing:** Encodes categorical mushroom data for modeling.
- **Model Selection:** Choose between SVM, Logistic Regression, and Random Forest.
- **Hyperparameter Tuning:** Adjust model parameters from the sidebar.
- **Metrics Visualization:** Plot Confusion Matrix, ROC Curve, and Precision-Recall Curve.
- **Show Raw Data:** Option to display the raw mushroom dataset.

## Getting Started

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- pandas, numpy, scikit-learn

Install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn
```

### Dataset

Place the `mushrooms.csv` dataset in the project root directory.  
You can download it from [Kaggle: Mushroom Classification](https://www.kaggle.com/uciml/mushroom-classification).

### Running the App

```bash
streamlit run app.py
```

Open the provided local URL in your browser to interact with the app.

## Usage

1. Select a classifier from the sidebar.
2. Adjust hyperparameters as desired.
3. Choose which metrics to plot.
4. Click **Classify** to train and evaluate the model.
5. Optionally, check "Show raw data" to view the dataset.

## Project Structure

```
.
├── app.py
├── mushrooms.csv
└── README.md
```

## License

This project is for educational purposes.

---