# Scaling Impact Dashboard

A Streamlit application that demonstrates how different feature scaling approaches affect machine learning model performance. Compare models trained with raw features, Min-Max normalization, and standardization.

## Features

- **4 Built-in Datasets**: Wine, Breast Cancer, California Housing, Diabetes
- **12 Models**: 6 classification and 6 regression algorithms
- **3 Scaling Options**: Raw (no scaling), Normalised (MinMaxScaler), Standardised (StandardScaler)
- **Proper Cross-Validation**: 5-fold CV with Pipeline to prevent data leakage
- **Comprehensive Metrics**: Accuracy, F1, RMSE, MAE, R2, and timing information
- **Interactive Visualization**: Bar charts comparing performance across scaling methods

## Requirements

- Python 3.11+ (required by scikit-learn 1.8.0)

## Windows Setup

Open Command Prompt or PowerShell in the project directory and run:

```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Usage

1. **Select Dataset**: Choose from Wine, Breast Cancer, California Housing, or Diabetes
2. **Select Models**: Pick which models to compare (defaults to all)
3. **Select Scaling Options**: Choose scaling methods to compare (defaults to all)
4. **Run Comparison**: Click the button to execute cross-validation experiments
5. **Analyze Results**: View the comparison table and bar chart

## Technical Notes

### Pipeline Prevents Data Leakage

This application uses scikit-learn's `Pipeline` to ensure proper handling of feature scaling during cross-validation:

```python
Pipeline([("scaler", StandardScaler()), ("model", SVC())])
```

**Why this matters:**
- The scaler is fit **only on the training fold** within each CV iteration
- The test fold is transformed using parameters learned from the training fold
- This prevents information from the test set leaking into the model training
- Without Pipeline, fitting a scaler on the entire dataset before CV would artificially inflate performance metrics

### Expected Behavior

**Scale-Sensitive Models** (typically improve with scaling):
- **KNN**: Distance calculations are dominated by high-magnitude features without scaling
- **SVM (RBF kernel)**: Kernel computations are sensitive to feature scales
- **Logistic Regression**: Gradient descent converges faster with scaled features
- **Neural Networks (MLP)**: Activation functions work best with normalized inputs

**Scale-Invariant Models** (little to no change with scaling):
- **Decision Trees**: Split decisions based on feature thresholds are scale-independent
- **Random Forests**: Ensemble of decision trees, inherits scale invariance

### Cross-Validation Strategy

- **Classification**: StratifiedKFold (preserves class distribution in each fold)
- **Regression**: KFold (standard k-fold splitting)
- Both use `n_splits=5`, `shuffle=True`, `random_state=42` for reproducibility

### Metrics

**Classification:**
- Accuracy (mean across folds)
- F1 Macro (mean across folds)

**Regression:**
- RMSE - Root Mean Squared Error (mean across folds)
- MAE - Mean Absolute Error (mean across folds)
- R2 - Coefficient of Determination (mean across folds)

**Timing (all tasks):**
- CV fit time (mean) - Average time to fit model across folds
- CV score time (mean) - Average time to score model across folds
- Total run time - Wall clock time for the entire cross_validate call

## Project Structure

```
model-perf/
├── app.py              # Main Streamlit application
├── requirements.txt    # Pinned package versions
└── README.md           # This file
```

## Dependencies

| Package | Version |
|---------|---------|
| streamlit | 1.53.0 |
| scikit-learn | 1.8.0 |
| numpy | 2.4.1 |
| scipy | 1.17.0 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| joblib | 1.5.3 |
| threadpoolctl | 3.6.0 |
