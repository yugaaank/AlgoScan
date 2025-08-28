from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import json
from algorithm import CryptoIdentifier

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Initialize the crypto identifier
crypto_identifier = CryptoIdentifier()

@app.route('/')
def index():
    """Main page with the analysis form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the provided hex data"""
    try:
        data = request.get_json()
        hex_input = data.get('hex_input', '').strip()
        
        if not hex_input:
            return jsonify({'error': 'Please provide hex data to analyze'}), 400
        
        # Validate that we have a trained model
        if crypto_identifier.model is None:
            return jsonify({'error': 'No trained model available. Please train a model first.'}), 400
        
        # Perform the analysis
        results = crypto_identifier.identify_algorithm(hex_input)
        
        # Format results for JSON response
        formatted_results = []
        for algo, confidence in results:
            level = 'HIGH' if confidence > 0.7 else ('MEDIUM' if confidence > 0.4 else 'LOW')
            formatted_results.append({
                'algorithm': algo,
                'confidence': round(confidence * 100, 2),
                'level': level
            })
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'input_length': len(hex_input)
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train a new model with the provided dataset"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', 'cryptography_dataset_generated.csv')
        
        if not os.path.exists(dataset_path):
            return jsonify({'error': f'Dataset file not found: {dataset_path}'}), 400
        
        # Train the model (this might take a while)
        crypto_identifier.train_model(dataset_path)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate the current model with a test dataset"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', 'cryptography_dataset_generated.csv')
        
        if not os.path.exists(dataset_path):
            return jsonify({'error': f'Dataset file not found: {dataset_path}'}), 400
        
        if crypto_identifier.model is None:
            return jsonify({'error': 'No trained model available for evaluation'}), 400
        
        # Capture the evaluation output
        import io
        import sys
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            crypto_identifier.test_model(dataset_path)
        
        evaluation_output = f.getvalue()
        
        return jsonify({
            'success': True,
            'evaluation_output': evaluation_output
        })
        
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get the current status of the model"""
    model_trained = crypto_identifier.model is not None
    dataset_exists = os.path.exists('cryptography_dataset_generated.csv')
    
    return jsonify({
        'model_trained': model_trained,
        'dataset_exists': dataset_exists,
        'available_algorithms': list(crypto_identifier.label_encoder.classes_) if crypto_identifier.label_encoder else []
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)