"""
WhatsApp Chat Analyzer - Flask Web Application
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from src.pipeline import ChatAnalysisPipeline
import pandas as pd


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Home page with upload form"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only .txt files allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Run analysis pipeline
        try:
            pipeline = ChatAnalysisPipeline(file_path)
            success = pipeline.run_full_pipeline()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Analysis complete!'
                }), 200
            else:
                return jsonify({'error': 'Analysis failed'}), 500
        
        except Exception as e:
            return jsonify({'error': f'Analysis error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500


@app.route('/results')
def show_results():
    """Show analysis results page"""
    try:
        df = pd.read_csv('sentiment_analyzed.csv')
        
        stats = {
            'total_messages': len(df),
            'unique_users': df['sender'].nunique() if 'sender' in df.columns else 0,
        }
        
        return render_template('results.html', stats=stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    try:
        df = pd.read_csv('sentiment_analyzed.csv')
        
        stats = {
            'total_messages': int(len(df)),
            'unique_users': int(df['sender'].nunique()) if 'sender' in df.columns else 0,
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)