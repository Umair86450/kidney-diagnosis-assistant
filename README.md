# 🩺 Kidney Disease Classification and Medical Report Generation

## 🚀 Project Overview

This project is an **AI-powered web application** designed to classify kidney-related medical images and generate a **comprehensive medical analysis report** based on user-input symptoms and history. It combines the power of **deep learning (TensorFlow)** for image classification and **Groq LLM** for natural language generation of the final report.

---

## ✨ Features

* 📤 Upload kidney medical images (e.g., ultrasound, X-ray).
* 🧠 Classifies images into one of four categories: **Cyst, Normal, Stone, Tumor**.
* 📝 Collects user inputs such as symptoms, pain level, history, and more.
* 📄 Automatically generates a detailed and structured **medical report**.
* 🌐 Intuitive, Flask-based web interface for ease of use.

---

## ⚙️ How It Works

### 1. **User  Interaction**

* The user visits the web app and uploads a kidney image.
* A short questionnaire is filled out with medical history and symptoms.

### 2. **Image Preprocessing**

* The image is resized and normalized to match the input requirements of the model.

### 3. **Kidney Condition Prediction**

* The pre-trained **TensorFlow model (`model.h5`)** predicts the kidney condition:
  * Cyst
  * Normal
  * Stone
  * Tumor

### 4. **Contextual Data Collection**

* Form responses provide additional context for the report generation.

### 5. **AI-Powered Report Generation**

* A structured report is created using **Groq’s LLM (e.g., Mixtral)**:
  * Includes predicted condition
  * Summary of patient symptoms
  * Medical implications and next steps
  * Lifestyle or treatment recommendations

### 6. **Results Display**

* The user is redirected to a result page with the full AI-generated report in a readable format.

### 7. **Robust Error Handling**

* Handles cases like missing files, bad input formats, or model/reporting errors gracefully.

---

## 🧠 Technologies Used

* **TensorFlow** – For model training and kidney disease classification
* **Groq API** – For structured report generation using large language models
* **Flask** – Lightweight backend web framework
* **Pillow & NumPy** – For image handling
* **HTML/CSS** – For front-end design and report formatting

---

## 📦 Dataset & Model Training

* The kidney disease image dataset is sourced from **[Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone))**.
* The model is trained offline and saved as `model.h5`.
* This model is loaded into the web app for real-time classification.

---

## 🔧 Installation & Setup

```bash
git clone https://github.com/YourUsername/kidney-disease-detection.git
cd kidney-disease-detection
pip install -r requirements.txt
python app.py
```

> Ensure you have your Groq API key configured before starting.

---

## 🛠️ Troubleshooting

* **API Errors**: Verify that your Groq API key is correctly set in the `.env` file. Without this key, the application will not run properly.
  
---

## 📄 Example Use Case

* A patient uploads a kidney scan.
* They answer basic medical questions.
* The app predicts a **“Stone”** condition.
* A detailed medical report is generated automatically, ready for review.

---

## ✅ Future Improvements

* Integration with electronic health records (EHR).
* Support for multiple image formats and larger models.
* Enable PDF download of the report.
* Add multilingual support.

---

## 🔗 Live Demo & GitHub

GitHub Repository: [Umair86450/automated-research-pipeline](https://github.com/Umair86450/automated-research-pipeline)

---

## 🔗 Demo

🎥 **Watch the Live Demo**  
Experience the full workflow in action:  
👉 [Click here to view the demo video](https://drive.google.com/file/d/1GH65NN_CoRtnNeK9Y5YYwS0nL75I4G5l/view?usp=sharing)

---
```

This README file now includes a troubleshooting section that emphasizes the importance of configuring the Groq API key for the application to run correctly. You can customize the GitHub repository link and any other specific details as needed.
