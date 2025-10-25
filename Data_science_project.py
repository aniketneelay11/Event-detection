import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

DATA_FILE = r'/home/aniket/Data science project/belle-II data.csv' #add file here
TARGET_COLUMN = 'type'
INDEX_COLUMN = 'Unnamed: 0'

def explore_belle2_data_info(filepath, target_col):

    # Load the Data
    try:
        df = pd.read_csv(filepath)
        print(f" Successfully loaded data from '{filepath}'.")
        print(f"Dataset contains {df.shape[0]} events and {df.shape[1]} columns.\n")
    except FileNotFoundError:
        print(f" ERROR: The file '{filepath}' was not found.")
        return None

    # Initial Data Inspection
    print("First 5 events in the dataset:")
    print(df.head()) 
    print("\nDataset information:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("-" * 50 + "\n")

    # Analyze Target Variable
    if target_col not in df.columns:
        print(f" ERROR: Target column '{target_col}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return df

    event_counts = df[target_col].value_counts().sort_index()
    print("Count of each event type:")
    print(event_counts)

    # Check for class imbalance
    imbalance_ratio = event_counts.std() / event_counts.mean()
    if imbalance_ratio >= 0.1:
        print("\n Warning: Dataset appears to be imbalanced.")
    else:
        print("\n Dataset appears to be reasonably balanced.")

    # Plot target column distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title('Distribution of Event Types', fontsize=16)
    plt.xlabel('Event Type Label', fontsize=12)
    plt.ylabel('Number of Events', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return df

df = explore_belle2_data_info(DATA_FILE,TARGET_COLUMN)

def explore_belle2_correlation(df, target_col):
    
    feature_columns = df.columns.drop(target_col)
    numeric_features = df[feature_columns].select_dtypes(include=np.number).columns.tolist()

    print(f"Found {len(numeric_features)} numeric features.")

    # Compute correlation matrix
    correlation_matrix = df[numeric_features].corr()

    # Show top correlated pairs
    corr_pairs = (
        correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
        .head(10)
    )
    print("Top 10 correlated feature pairs:\n", corr_pairs, "\n")

    # Plot correlation heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap of Features', fontsize=16)
    plt.show()

explore_belle2_correlation(df,TARGET_COLUMN)
def process_data_for_binary_task(filepath, target_col, index_col=None):

    # Load Data
    try:
        df = pd.read_csv(filepath)
        print(f" Successfully loaded data from '{filepath}'.\n")
    except FileNotFoundError:
        print(f" ERROR: The file '{filepath}' was not found.")
        return None, None, None, None

    # Drop the index column if it exists
    if index_col and index_col in df.columns:
        df = df.drop(columns=[index_col])
        print(f"Dropped index column: '{index_col}'")

    # --- Convert to Binary Classification Task ---
    print("Transforming target variable into a binary problem...")
    original_counts = df[target_col].value_counts().sort_index()

    df[target_col] = df[target_col].apply(lambda x: 0 if x in [0, 1] else 1)

    new_counts = df[target_col].value_counts().sort_index()
    print("Original event counts:\n", original_counts)
    print("\nNew binary event counts (Class 0: [0,1], Class 1: [2,3,4,5]):\n", new_counts)
    print("-" * 40)

    # Separate Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"Original number of features: {X.shape[1]}\n")

    # Split into Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(" Features scaled using StandardScaler.\n")

    # PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA selected {pca.n_components_} components to explain 95% of variance.")

    # Show Top Features per PC
    print("\n--- Top Contributing Features for each Principal Component ---")
    feature_names = X.columns
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC_{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )

    for pc in loadings_df.columns:
        top_features = loadings_df[pc].abs().sort_values(ascending=False).head(5)
        print(f"\nTop 5 features for {pc}:")
        print(top_features)
    print("\n" + "-" * 40)

    print(f"Original training data shape: {X_train.shape}")
    print(f"PCA-transformed training data shape: {X_train_pca.shape}")

    return X_train_pca, X_test_pca, y_train, y_test


# --- Run PCA ---
X_train_pca, X_test_pca, y_train, y_test = process_data_for_binary_task(DATA_FILE, TARGET_COLUMN, INDEX_COLUMN)

def evaluate_model(model_name, model, X_test, y_test):
    print(f"--- Evaluating Model: {model_name} ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy:.4f}")

    # Calculate and display ROC AUC Score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"Area Under ROC Curve (AUC): {auc_score:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1]))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.show()

    # Plot the ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Logistic Regression (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test):
    """Trains and evaluates a Logistic Regression model."""
    print("\n--- Starting Model Training and Evaluation ---")

    print("\nTraining Logistic Regression model...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_pca, y_train)
    evaluate_model("Logistic Regression", log_reg, X_test_pca, y_test)

if X_train_pca is not None:
    train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test)
