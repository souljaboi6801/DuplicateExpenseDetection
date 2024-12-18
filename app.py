from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uuid

# Add Tesseract Path for Windows (update as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
SAVED_TEXT_FOLDER = 'saved_texts/'
FLAGGED_DUPLICATES_FOLDER = 'flagged_duplicates/'
MODEL_DIR = os.path.abspath(r"C:\Users\Nitesh Kumar\Downloads\fine_tuned_tinybert")  # Local model directory

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVED_TEXT_FOLDER, exist_ok=True)
os.makedirs(FLAGGED_DUPLICATES_FOLDER, exist_ok=True)

# Verify the model directory
required_files = ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"]
for file in required_files:
    if not os.path.exists(os.path.join(MODEL_DIR, file)):
        raise FileNotFoundError(f"Missing required file: {file} in {MODEL_DIR}")

# Load the tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret_key'

# Mock user data for login/signup
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

# OCR Function to extract text from image or PDF
def extract_text(file_path):
    text = ""
    if file_path.endswith('.pdf'):
        images = convert_from_path(file_path)
        for image in images:
            text += pytesseract.image_to_string(image)
    else:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    return text

# Save extracted text locally
def save_extracted_text(text, file_name, user_name):
    unique_id = str(uuid.uuid4())
    text_path = os.path.join(SAVED_TEXT_FOLDER, f"{unique_id}.json")
    with open(text_path, 'w') as f:
        json.dump({"id": unique_id, "content": text, "file_name": file_name, "uploaded_by": user_name}, f)


# Save flagged duplicates
def save_flagged_duplicate(text, original_file, duplicate_file, confidence, user_name):
    unique_id = str(uuid.uuid4())
    duplicate_info = {
        "id": unique_id,
        "content": text,
        "original_file": original_file,
        "duplicate_file": duplicate_file,
        "confidence": confidence,
        "uploaded_by": user_name
    }
    flagged_path = os.path.join(FLAGGED_DUPLICATES_FOLDER, f"{unique_id}.json")
    with open(flagged_path, 'w') as f:
        json.dump(duplicate_info, f)

# Load all previously saved texts
def load_saved_texts():
    texts = []
    for file in os.listdir(SAVED_TEXT_FOLDER):
        if file.endswith('.json'):
            with open(os.path.join(SAVED_TEXT_FOLDER, file)) as f:
                data = json.load(f)
                data.setdefault("file_name", os.path.splitext(file)[0])  # Add 'file_name' if missing
                texts.append(data)
    return texts

# Function to predict duplicate or not
def predict_duplicate(text1, text2):
    inputs = tokenizer(
        text1, text2,
        return_tensors="pt", truncation=True,
        padding="max_length", max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    confidence, predicted_label = torch.max(probabilities, dim=1)
    label = "Duplicate" if predicted_label.item() == 1 else "Not Duplicate"
    return label, confidence.item()

def find_duplicates(new_text, current_file, user_name):
    saved_texts = load_saved_texts()
    duplicates = []
    for saved_entry in saved_texts:
        saved_text = saved_entry["content"]
        original_file = saved_entry["file_name"]

        # Skip comparison if the current file is the same as the original file
        if current_file == original_file:
            continue

        # Predict duplication using the model
        label, confidence = predict_duplicate(new_text, saved_text)

        # Additional validation
        if label == "Duplicate" and confidence > 0.8:
            # Save flagged duplicate
            save_flagged_duplicate(new_text, original_file, current_file, confidence, user_name)
            duplicates.append((original_file, confidence))
    return duplicates


# Login required decorator
def login_required(func):
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/flagged_duplicates', methods=['GET', 'POST'])
@login_required
def flagged_duplicates():
    if session.get('role') != 'admin':
        flash("Admins only!")
        return redirect(url_for('home'))

    flagged_reports = [json.load(open(os.path.join(FLAGGED_DUPLICATES_FOLDER, f))) 
                       for f in os.listdir(FLAGGED_DUPLICATES_FOLDER)]
    return render_template('flagged_duplicates.html', reports=flagged_reports)


# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['user'] = username
            session['role'] = users[username]['role']
            flash('Login successful!', 'success')  # Green popup for success
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials, please try again.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if not username or not password or not confirm_password:
            flash('All fields are required!', 'error')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        if username in users:
            flash('Username already exists!', 'error')
            return redirect(url_for('signup'))

        users[username] = {'password': password, 'role': 'user'}
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Extract text from the uploaded file
            extracted_text = extract_text(file_path)

            # Detect duplicates and pass the logged-in user as a parameter
            duplicates = find_duplicates(extracted_text, file.filename, session['user'])

            # Save extracted text along with user information
            save_extracted_text(extracted_text, file.filename, session['user'])

            if duplicates:
                flash(f"Duplicates Found: {duplicates}")
            else:
                flash("No duplicates found.")
    return render_template('upload.html')

@app.route('/home', methods=['GET'])
@login_required
def home():
    username = session.get('user')  # Get the logged-in username
    role = session.get('role')      # Get the user role
    return render_template('home.html', username=username, role=role)


@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_page():
    if session.get('role') != 'admin':
        flash("Admins only!")
        return redirect(url_for('home'))

    # Handle POST actions for Accept/Delete
    if request.method == 'POST':
        action = request.form['action']
        file_id = request.form['file_id']
        flagged_file_path = os.path.join(FLAGGED_DUPLICATES_FOLDER, f"{file_id}.json")

        if os.path.exists(flagged_file_path):
            if action == "accept":
                # Remove the duplicate file's text from saved_texts
                with open(flagged_file_path) as f:
                    report = json.load(f)
                    duplicate_file = report["duplicate_file"]

                # Remove duplicate from saved_texts
                saved_text_file = os.path.join(SAVED_TEXT_FOLDER, f"{file_id}.json")
                if os.path.exists(saved_text_file):
                    os.remove(saved_text_file)

                # Remove uploaded duplicate image
                uploaded_file_path = os.path.join(UPLOAD_FOLDER, duplicate_file)
                if os.path.exists(uploaded_file_path):
                    os.remove(uploaded_file_path)

                flash(f"Duplicate {duplicate_file} accepted and data cleaned.")
            elif action == "delete":
                # Delete flagged report file
                os.remove(flagged_file_path)
                flash(f"Report {file_id} deleted successfully.")

    # Load flagged reports
    flagged_reports = []
    for file in os.listdir(FLAGGED_DUPLICATES_FOLDER):
        with open(os.path.join(FLAGGED_DUPLICATES_FOLDER, file)) as f:
            flagged_reports.append(json.load(f))

    return render_template('admin.html', reports=flagged_reports)

if __name__ == '__main__':
    app.run(debug=True)
