import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import traceback

# Import your existing prediction logic
try:
    from predict import get_prediction
    import config
except ImportError:
    print("FATAL ERROR: Make sure predict.py and config.py are present.")
    exit()

# Configure the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024
app.secret_key = 'a_very_secret_key_for_flask_sessions' 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'mp3', 'wav', 'flac'}

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """Handles the file upload and runs the prediction model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)

            all_results = get_prediction(filepath)
            
            CONFIDENCE_THRESHOLD = 0.10  # 10%
            
            confident_results = [res for res in all_results if res[1] >= CONFIDENCE_THRESHOLD]
            sorted_results = sorted(confident_results, key=lambda item: item[1], reverse=True)

            if not sorted_results:
                final_predictions = sorted(all_results, key=lambda item: item[1], reverse=True)[:1]
            else:
                final_predictions = sorted_results[:3]

            # The frontend will now handle all visual highlighting based on these results.
            formatted_results = []
            for code, prob in final_predictions:
                label_name = config.LABEL_MAP.get(code, "Unknown")
                formatted_results.append({
                    "name": label_name,
                    "confidence": f"{prob:.2%}"
                })
            
            return jsonify({'results': formatted_results, 'filename': filename})

        except Exception:
            app.logger.error(f"Prediction failed for {filename}: {traceback.format_exc()}")
            return jsonify({'error': 'An internal error occurred during analysis.'}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Access your application at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

