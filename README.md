# AlgoScan
AlgoScan is a web-based tool for automatic identification of cryptographic algorithms from raw hex data. Powered by machine learning (CatBoost), it analyzes ciphertext or hash data and predicts the underlying algorithm, offering easy training, evaluation, and analysis via a Flask web interface.

Features
Automatic hex data analysis: Predicts the cryptographic algorithm used.

Model training and evaluation: Train with your own dataset, get model accuracy and performance stats.

Easy-to-use web interface: Upload hex data, run analysis, and view results from your browser.

Supports RSA, SHA256, and other learned algorithms.

Robust feature extraction pipeline: Includes statistical, block, and temporal pattern analysis.

Requirements
Python 3.9 or later

The following Python packages:

Flask==2.3.3

numpy==1.24.3

pandas==2.0.3

scipy==1.11.1

scikit-learn==1.3.0

catboost==1.2

joblib==1.3.1

A training dataset in CSV format (default: cryptography_dataset_generated.csv).

Setup
Clone the repository.

Install dependencies:

bash
pip install -r requirements.txt
(Optional) Prepare your dataset:

The dataset CSV should include two columns: Ciphertext (hex strings) and Algorithm (algorithm label).

Save it as cryptography_dataset_generated.csv or specify your path during training.

Running the Application
Start the web interface using the included runner script:

bash
python run.py
This launches the Flask app locally on port 5000 (http://localhost:5000).[4]

First-Time Model Training
If no trained model is found, the app will prompt you to train one:

Go to /train via the web interface or POST to the /train endpoint with your dataset path (JSON):

json
{
  "dataset_path": "cryptography_dataset_generated.csv"
}
Training will extract features, train the CatBoost model, and save all artifacts.

Analyzing Data
Use the main interface to paste hex data and submit for analysis.

Alternatively, POST to /analyze:

json
{
  "hex_input": "YOUR_HEX_DATA_HERE"
}
The response includes probable algorithms, confidence scores, and classification level.

Evaluating the Model
To check performance on a test set, POST to /evaluate:

json
{
  "dataset_path": "cryptography_dataset_generated.csv"
}
The response shows accuracy and detailed classification metrics.

Checking Status
GET /status for details about model readiness and available algorithms.

File Structure
File	Purpose
algorithm.py	Main feature extraction, training, and model logic 
app.py	Flask web application endpoints 
run.py	Application runner (entry point) 
requirements.txt	Python dependencies 
README.md	This documentation
Usage Notes
For best results, use longer hex strings (ciphertext or hash outputs).

Training may take several minutes for large datasets.

All models are persisted as .pkl files in the working directory.

License
This project is released under the MIT License.
