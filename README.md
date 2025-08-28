# AlgoScan: Cryptographic Algorithm Identification from Hex Data

AlgoScan is a powerful, web-based tool for automatically identifying cryptographic algorithms from raw hexadecimal data. Leveraging a machine learning model built with **CatBoost**, it analyzes ciphertext or hash data to predict the underlying algorithm. This tool offers an intuitive web interface for easy training, evaluation, and analysis.

-----

## Features

  * **Automatic Hex Data Analysis**: Predicts the cryptographic algorithm used in a given hex string.
  * **Model Training and Evaluation**: Train the CatBoost model with your own datasets and get detailed performance metrics and accuracy.
  * **User-Friendly Web Interface**: A simple Flask-based web interface allows you to upload hex data, run analyses, and view results directly in your browser.
  * **Algorithm Support**: The model supports various cryptographic algorithms, including **RSA** and **SHA256**, and can be trained to recognize others.
  * **Robust Feature Extraction**: The tool uses a sophisticated feature extraction pipeline that includes statistical, block, and temporal pattern analysis to ensure high accuracy.

-----

## Requirements

  * **Python 3.9** or later
  * The following Python packages, which can be installed from `requirements.txt`:
      * `Flask==2.3.3`
      * `numpy==1.24.3`
      * `pandas==2.0.3`
      * `scipy==1.11.1`
      * `scikit-learn==1.3.0`
      * `catboost==1.2`
      * `joblib==1.3.1`
  * A training dataset in **CSV format** with two columns: `Ciphertext` (hex strings) and `Algorithm` (algorithm labels). A default dataset, `cryptography_dataset_generated.csv`, is expected.

-----

## Setup and Installation

1.  Clone the repository:

    ```bash
    git clone [repository-url]
    cd algoscan
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  (Optional) Prepare your dataset:

      * Create a CSV file with the columns `Ciphertext` and `Algorithm`.
      * Save it as `cryptography_dataset_generated.csv` in the root directory, or specify a different path during model training.

-----

## Usage

### Running the Application

To start the web application, run the `run.py` script from your terminal:

```bash
python run.py
```

This will launch the Flask application locally on port `5000`. You can access it in your browser at `http://localhost:5000`.

### First-Time Model Training 🧠

If no trained model exists, the application will prompt you to train one. You can initiate training through the web interface or via an API call.

**Web Interface**: Navigate to `/train`.

**API Endpoint**: Send a `POST` request to the `/train` endpoint with a JSON body specifying your dataset path:

```json
{
    "dataset_path": "cryptography_dataset_generated.csv"
}
```

The training process involves feature extraction, model training, and saving all artifacts as `.pkl` files in the working directory. This may take a few minutes for large datasets.

### Analyzing Data

You can analyze hex data by pasting it into the main web interface or by using the `/analyze` API endpoint.

**Web Interface**: Paste your hex data into the input field and click "Submit."

**API Endpoint**: Send a `POST` request to `/analyze` with a JSON body:

```json
{
    "hex_input": "YOUR_HEX_DATA_HERE"
}
```

The response will include the most probable algorithm, its confidence score, and the classification level. For best results, use longer hex strings (ciphertext or hash outputs).

### Evaluating the Model

To evaluate the model's performance on a test set, use the `/evaluate` endpoint.

**API Endpoint**: Send a `POST` request to `/evaluate` with a JSON body:

```json
{
    "dataset_path": "cryptography_dataset_generated.csv"
}
```

The response will provide the model's overall accuracy and detailed classification metrics.

### Checking Application Status

To check if a model is ready and see the list of algorithms it supports, send a `GET` request to the `/status` endpoint:

```bash
GET /status
```

-----

## File Structure

| File                     | Purpose                                          |
| :----------------------- | :----------------------------------------------- |
| `algorithm.py`           | Handles all feature extraction, training, and model logic. |
| `app.py`                 | Contains the Flask web application endpoints.    |
| `run.py`                 | The main application runner and entry point.     |
| `requirements.txt`       | Lists all Python dependencies.                   |
| `README.md`              | This documentation file.                         |
| `cryptography_dataset_generated.csv` | Default dataset for training (user-provided). |
| `*.pkl`                  | Persisted model artifacts (generated after training). |

-----

## License

This project is released under the **MIT License**. For more details, see the `LICENSE` file.
