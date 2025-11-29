# Parkinsons-Disease-ML-Predictive-Model-Webpage

A lightweight web application that provides **machine-learning–based predictions for Parkinson’s disease** using patient data with a **97% of accuracy**.  
The project integrates a trained ML model with an accessible web interface, making it ideal for demonstration, research exploration, and educational use. [**VISIT THE WEBPAGE HERE**](https://parkinsons-disease-ml-predictive-model-webpage.streamlit.app/") 

> ⚠️ **Disclaimer**: This tool is NOT intended for clinical diagnosis. It is for research and prototyping purposes only.
      
*You can explore the exploratory data analysis, model development and presentation in this [github repository](https://github.com/LunaPerezT/Parkinson-s-Disease-Predictive-ML-Model/tree/main/docs)*    
    
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
Contains raw, processed, datasets or testing data used for model inference or demonstration inside the web interface.

#### [`/models`](./models)
Stores trained machine-learning models in Pickle format(`.pkl`).   
   
*You can explore the exploratory data analysis, model development and presentation in this [github repository](https://github.com/LunaPerezT/Parkinson-s-Disease-Predictive-ML-Model/tree/main/docs)*

#### [`/img`](./img)
Images displayed within the webpage (logos, UI elements, figures).

#### [`/static`](./static)
Font collection as static frontend assets.

#### [`/.streamlit`](./.streamlit)  
Configuration options of the app defined in a `config.toml` file.

### 📄 Files

#### [`app.py`](./app.py)
Main application script.  
Runs the web interface (e.g., Streamlit/Flask) and loads the ML model for predictions.

#### [`requirements.txt`](./requirements.txt)
Python dependencies required to run the project.

Install with:
```bash
pip install -r requirements.txt
```
   
---
   
## 🚀 Getting Started

Follow these steps on your git bash to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/LunaPerezT/Parkinsons-Disease-ML-Predictive-Model-Webpage.git
cd Parkinsons-Disease-ML-Predictive-Model-Webpage
```
   
### 2. (Optional) Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies
    
```bash
pip install -r requirements.txt
```
    
### 4. Run the Web Application
   
```bash
python app.py
```

##🙋🏻‍♀️ About the Author
    
I’m **Luna Pérez Troncoso**, a **Data Scientist** with a strong foundation in **Artificial Intelligence, data analytics, and computational modeling**, originally shaped through my background in **science and research**. Over time, I’ve transitioned my **analytical and experimental mindset** into the tech field, where I design **data-driven solutions** that **transform complex information into actionable insights**.
         
I have experience working across the **full data pipeline**, from data acquisition, cleaning, and exploration to building, validating, and deploying predictive models. My focus is on **leveraging machine learning and statistical methods to uncover patterns, optimize processes, and support strategic decisions**.
           
What defines my approach is a balance between **technical precision and creativity**. I’m passionate about **connecting raw data with real-world** impact, collaborating with **cross-functional teams**, and communicating insights in a way that drives innovation.
<a href="https://www.linkedin.com/in/luna-p%C3%A9rez-troncoso-0ab21929b/">
 <img src="./img/linkedin.jpg" alt="linkedin" width="200"/>
</a>
Visit my [linkedIn profile](https://www.linkedin.com/in/luna-p%C3%A9rez-troncoso-0ab21929b/)

