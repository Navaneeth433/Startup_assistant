from flask import Flask, render_template, request, jsonify
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import tempfile
import os
import ollama

app = Flask(__name__)

# If needed, set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to extract text from images or PDFs
def extract_text_from_file(file_path, file_ext):
    text = ""
    try:
        if file_ext.lower() == ".pdf":
            # Convert PDF pages to images
            pages = convert_from_path(
                file_path, 
                poppler_path=r"C:\poppler-windows-25.07.0-0\poppler-25.07.0\Library\bin"  # <-- update with your actual Poppler bin path
            )
            for page in pages:
                text += pytesseract.image_to_string(page)
        else:
            # Process as image
            text = pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        print(f"Error extracting text: {e}")
        text = ""
    return text.strip()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    # Save temporarily
    suffix = os.path.splitext(file.filename)[1]  # preserve file extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        file_path = tmp.name

    # Extract text using OCR / PDF conversion
    file_ext = os.path.splitext(file.filename)[1]
    text = extract_text_from_file(file_path, file_ext)

    if not text:
        os.remove(file_path)
        return jsonify({'error': 'No text detected. Try a clearer scan or a text-based PDF.'})

    # Summarize using Mistral via Ollama
    try:
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": "You are a legal assistant. Summarize the given legal text clearly and concisely."},
                {"role": "user", "content": text}
            ]
        )
        summary = response['message']['content']
    except Exception as e:
        summary = f"❌ Error contacting Ollama: {e}"

    # Clean up temp file
    os.remove(file_path)
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True)
