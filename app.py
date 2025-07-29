from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load the summarization model once when the app starts
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text', '')
    max_length = data.get('max_length', 150)
    min_length = data.get('min_length', 30)

    if not text:
        return jsonify({'error': 'No text provided for summarization'}), 400

    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return jsonify({'summary': summary[0]['summary_text']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
