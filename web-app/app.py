# # Define the multiple-choice questions
# multiple_choice_questions = [
#     {
#         "question": "Do you experience pain in your kidney area?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Have you had any previous kidney issues?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Do you have a history of kidney stones?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Have you noticed any blood in your urine?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Do you experience frequent urination?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Have you had any recent infections?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Do you have high blood pressure?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Have you undergone any kidney surgeries?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Do you experience swelling in your legs or ankles?",
#         "choices": ["Yes", "No"]
#     },
#     {
#         "question": "Have you been diagnosed with diabetes?",
#         "choices": ["Yes", "No"]
#     }



# ==============================================================================================================
# Finalize Code
# ===========================================================================================================


import os
import logging
from flask import Flask, request, render_template, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path
from groq import Groq  # Ensure Groq is properly installed and configured

# ============================
# Configuration and Setup
# ============================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Define label mapping consistent with training
labels = ["Cyst", "Normal", "Stone", "Tumor"]

# ============================
# Model Loading Function
# ============================

def load_model():
    try:
        model_path = Path("model.h5")
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}.")
            return None
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as ex:
        logger.error(f"Error loading the model: {ex}")
        return None

# Load the Model at Startup
model = load_model()
if model is None:
    logger.error("Failed to load the model. Exiting application.")
    exit(1)

# ============================
# Groq Client Initialization
# ============================

import os
from dotenv import load_dotenv
# baaki imports...

# Load environment variables from .env
load_dotenv()

# Initialize Groq client with API key from env var
def initialize_groq():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found in environment variables.")
            return None
        client = Groq(api_key=api_key)
        logger.info("Groq client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Error initializing Groq client: {e}")
        return None

client = initialize_groq()




# ============================
# Image Preprocessing Function
# ============================

def preprocess_image(image):
    image = tf.image.resize(image, [28, 28])  # Resize to 28x28
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image / 255.0  # Normalize to [0, 1]

# ============================
# Flask Routes
# ============================

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            logger.warning("No file part in the request.")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file.")
            return jsonify({'error': 'No selected file'}), 400

        try:
            # Process image and get prediction
            image = Image.open(file).convert('RGB')
            image_array = preprocess_image(tf.convert_to_tensor(np.array(image)))

            # Make prediction
            predictions = model.predict(image_array)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            kidney_type = labels[predicted_idx]

            # Collect form data
            form_data = {
                'pain': request.form.get('pain', 'No'),
                'history': request.form.get('history', 'No'),
                'pain_level': request.form.get('pain_level', 'No Pain'),
                'pain_duration': request.form.get('pain_duration', 'No Pain'),
                'symptoms': request.form.get('symptoms', 'None'),
                'family_history': request.form.get('family_history', 'No')
            }

            # Create structured report content
            chat_context = f"""
            <div style="font-family: 'Arial', sans-serif; line-height: 1.6; color: #333; margin: 20px;">
                <h1 style="text-align: center; color: #2C3E50; font-size: 2.5rem; margin-bottom: 20px;">Medical Analysis Report</h1>

                <div style="border: 1px solid #00796B; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
                    <h2 style="color: #00796B; font-size: 1.8rem;">Patient Information</h2>
                    <p style="font-size: 1.2rem;">
                        <strong>Predicted Kidney Condition:</strong> <span style="color: #FF6F61; font-weight: bold;">{kidney_type}</span>
                    </p>
                </div>

                <div style="margin-top: 20px; border: 1px solid #00796B; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
                    <h2 style="color: #00796B; font-size: 1.8rem;">Image Analysis</h2>
                    <p style="font-size: 1.2rem;">
                        <strong>Initial Diagnosis:</strong> The kidney images suggest a possible condition based on image processing.
                    </p>
                </div>

                <div style="margin-top: 20px; border: 1px solid #00796B; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
                    <h2 style="color: #00796B; font-size: 1.8rem;">Patient Symptoms and History</h2>
                    <ul style="padding-left: 20px; font-size: 1.2rem;">
                        <li><strong>General Pain:</strong> {form_data['pain']}</li>
                        <li><strong>Known Kidney History:</strong> {form_data['history']}</li>
                        <li><strong>Pain Level:</strong> {form_data['pain_level']}</li>
                        <li><strong>Pain Duration:</strong> {form_data['pain_duration']}</li>
                        <li><strong>Additional Symptoms:</strong> {form_data['symptoms']}</li>
                        <li><strong>Family History of Kidney Problems:</strong> {form_data['family_history']}</li>
                    </ul>
                </div>

                <div style="margin-top: 20px; border: 1px solid #00796B; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
                    <h2 style="color: #00796B; font-size: 1.8rem;">Comprehensive Medical Analysis</h2>
                    <ol style="padding-left: 20px; font-size: 1.2rem;">
                        <li><strong>Detailed Findings:</strong> Analysis based on the predicted type and patient symptoms.</li>
                        <li><strong>Implications of Diagnosis:</strong> Potential health implications related to the identified condition.</li>
                        <li><strong>Recommended Next Steps:</strong> Next steps for further evaluation and tests.</li>
                        <li><strong>Risk Factors to Consider:</strong> Identify risk factors that may need monitoring.</li>
                        <li><strong>Lifestyle Recommendations:</strong> Suggestions to improve kidney health.</li>
                    </ol>
                </div>

                <div style="margin-top: 20px; border: 1px solid #00796B; border-radius: 10px; padding: 20px; background-color: #f9f9f9;">
                    <h2 style="color: #00796B; font-size: 1.8rem;">Conclusion</h2>
                    <p style="font-size: 1.2rem;">
                        It's essential to consult a healthcare professional for personalized medical advice.
                    </p>
                </div>
            </div>
            """
            # Generate report using Groq
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a medical analysis assistant specializing in kidney conditions. Provide a structured report with clear headings and professional language."
                                "Include comprehensive details for each section, and make sure the language is professional and informative."
                            )
                        },
                        {
                            "role": "user",
                            "content": chat_context
                        }
                    ],
                    model="mixtral-8x7b-32768",
                    temperature=0.7,
                    max_tokens=10000
                )
                report = chat_completion.choices[0].message.content
                logger.info("Report generated successfully")
            except Exception as groq_error:
                logger.error(f"Error generating report: {groq_error}")
                report = "Error generating detailed report. Please consult with a healthcare professional."

            return redirect(url_for('results', kidney_type=kidney_type, report=report))

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return jsonify({'error': 'An error occurred during processing.', 'details': str(e)}), 500

    return render_template('index.html')


@app.route('/results')
def results():
    kidney_type = request.args.get('kidney_type', 'Unknown')
    report = request.args.get('report', 'No report available.')
    formatted_report = report.replace('\n', '<br>')
    
    return render_template('results.html', 
                           kidney_type=kidney_type, 
                           report=formatted_report)

# ============================
# Main Execution
# ============================

if __name__ == '__main__':
    if model:
        app.run(debug=True)
    else:
        logger.error("Model not loaded. Flask app will not start.")
