# WiFi Congestion Prediction Model

This project provides a complete pipeline to analyze WiFi traffic from `.pcap` files and train a machine learning model to predict network congestion in real-time.

## How It Works

The pipeline consists of two main stages:
1.  **Feature Extraction**: It processes a `.pcap` file, analyzing packets in one-second intervals. For each interval, it extracts key features like packet count, number of unique devices, and the distribution of WiFi frame types (Management, Control, Data).
2.  **Model Training**: It uses the extracted features to train a `RandomForestClassifier` model. This model learns the patterns associated with "Normal" and "Congested" network traffic.

## Requirements

*   Python 3.8+
*   The following Python libraries:
    *   `scapy`
    *   `pandas`
    *   `scikit-learn`
    *   `joblib`
    *   `seaborn`
    *   `matplotlib`

## Installation

1.  Clone the repository.
2.  Install the required libraries using pip:
    ```bash
    pip install scapy pandas scikit-learn joblib seaborn matplotlib
    ```

## Usage

There are two ways to run this pipeline.

### Option 1: Using Your Own `traffic.pcap` File

This is the standard workflow.

1.  **Place your data file**: Put your own packet capture file named `traffic.pcap` in the root directory of this project.
2.  **Run the feature extraction script**:
    ```bash
    python3 process_pcap.py
    ```
    This will generate a `features.csv` file.
3.  **Run the model training script**:
    ```bash
    python3 train_model.py
    ```
    This will train the model on your data and save the result.

### Option 2: Using the Synthetic Data Generator

If you don't have a `.pcap` file, you can generate a sample one for demonstration purposes.

1.  **Run the data generation script**:
    ```bash
    python3 generate_pcap.py
    ```
    This creates a sample `traffic.pcap` file with simulated normal and congested periods.
2.  **Follow steps 2 and 3** from "Option 1" to process the data and train the model.

## Output Files

After running the pipeline, the following files will be generated:
*   `traffic.pcap`: The raw packet data (either your own or synthetically generated).
*   `features.csv`: The processed data containing the features for each one-second time window.
*   `congestion_model.joblib`: The final, trained machine learning model, ready to be used for predictions.
*   `confusion_matrix.png`: A plot visualizing the performance of the trained model on the test data.
