# Parkinsons-Disease-ML-Predictive-Model-Webpage

A lightweight web application that provides **machine-learning–based predictions for Parkinson’s disease** using patient data with a **97% of accuracy**.  
The project integrates a trained ML model with an accessible web interface, making it ideal for demonstration, research exploration, and educational use.

[***VISIT THE WEBPAGE HERE**](https://parkinsons-disease-ml-predictive-model-webpage.streamlit.app/)

> ⚠️ **Disclaimer**: This tool is NOT intended for clinical diagnosis. It is for research and prototyping purposes only.

---

## 🌐 Repository Structure Overview

Parkinsons-Disease-ML-Predictive-Model-Webpage/
├── data/
├── models/
├── img/
├── static/
├── app.py
├── requirements.txt
└── README.md


Below is the structure with links and descriptions:

### 📁 Folders

#### [`/data`](./data)
Contains example datasets or testing data used for model inference or demonstration inside the web interface.

#### [`/models`](./models)
Stores trained machine-learning models (e.g., `.pkl`, `.sav`, etc.).

#### [`/img`](./img)
Images displayed within the webpage (logos, UI elements, figures).

#### [`/static`](./static)
Static frontend assets such as CSS, JavaScript, icons, and fonts.

---

### 📄 Files

#### [`app.py`](./app.py)
Main application script.  
Runs the web interface (e.g., Streamlit/Flask) and loads the ML model for predictions.

#### [`requirements.txt`](./requirements.txt)
Python dependencies required to run the project.

Install with:
```bash
pip install -r requirements.txt
