from flask import Flask, render_template, request, jsonify, Response
import os
import subprocess
import sys
from algorithm import CryptoIdentifier

app = Flask(__name__)
app.secret_key = os.urandom(24)

MODEL_FILE = 'crypto_model.pkl'

# Initialize the crypto identifier
crypto_identifier = CryptoIdentifier()

@app.route('/')
def index():
    """Main page with the analysis form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the provided hex data, or indicate that training is needed."""
    try:
        # Check the file on disk as the source of truth
        if not os.path.exists(MODEL_FILE):
            return jsonify({'action': 'start_training'})

        # If we get here, the model file exists.
        # Let's ensure the global identifier has it loaded.
        if crypto_identifier.model is None:
            print("Model file exists, but not loaded in memory. Loading now.")
            crypto_identifier.load_model()
            if crypto_identifier.model is None:
                # This is an unexpected state, indicating a problem with the model file itself.
                return jsonify({'error': 'Model file found but could not be loaded. Check server logs for details.'}), 500

        # Proceed with analysis
        data = request.get_json()
        hex_input = data.get('hex_input', '').strip()

        if not hex_input:
            return jsonify({'error': 'Please provide hex data to analyze'}), 400

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

@app.route('/start_training')
def start_training():
    """Starts the training process and streams the output."""
    def training_stream():
        # Use subprocess to call a training script and capture its output in real-time
        process = subprocess.Popen(
            [sys.executable, '-u', 'run_training.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        for line in iter(process.stdout.readline, ''):
            yield f"data: {line.strip()}\n\n"
        
        process.wait()
        
        # After training, reload the model for the current process to use immediately
        crypto_identifier.load_model()
        
        # Send a special message to the client to signal completion
        yield "data: __TRAINING_COMPLETE__\n\n"

    return Response(training_stream(), mimetype='text/event-stream')

@app.route('/status')
def status():
    """Get the current status of the model"""
    model_trained = os.path.exists(MODEL_FILE)
    dataset_exists = os.path.exists('cryptography_dataset_generated.csv')

    # Try to load the label encoder to get the list of algorithms if model is trained
    available_algorithms = []
    if model_trained:
        if crypto_identifier.label_encoder is None:
            crypto_identifier.load_model()
        if crypto_identifier.label_encoder is not None:
            available_algorithms = list(crypto_identifier.label_encoder.classes_)

    return jsonify({
        'model_trained': model_trained,
        'dataset_exists': dataset_exists,
        'available_algorithms': available_algorithms
    })




if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
