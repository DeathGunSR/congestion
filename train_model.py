import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_FILE = 'features.csv'
MODEL_OUTPUT_FILE = 'congestion_model.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Main Training Logic ---

def train_model():
    """
    Loads data, trains a classifier, evaluates it, and saves the model.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found.")
        print("Please run the feature extraction script first.")
        return

    print("Data loaded successfully.")
    print(f"Shape of the dataset: {df.shape}")

    # 2. Prepare Data
    # Drop the timestamp as it's not a direct feature for the model
    df = df.drop('timestamp', axis=1)

    # Define features (X) and target (y)
    X = df.drop('is_congested', axis=1)
    y = df['is_congested']

    feature_names = X.columns.tolist()
    print(f"Features being used: {feature_names}")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # 4. Train Model
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate Model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Congested']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Congested'], yticklabels=['Normal', 'Congested'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix plot saved to 'confusion_matrix.png'")

    # Feature Importance
    feature_importances = pd.DataFrame(model.feature_importances_, index=feature_names, columns=['importance']).sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(feature_importances)


    # 6. Save Model
    print(f"\nSaving model to '{MODEL_OUTPUT_FILE}'...")
    joblib.dump(model, MODEL_OUTPUT_FILE)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()
