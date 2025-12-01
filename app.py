# Streamlit Parkinson's Disease Predictive Machine Learning Model web page
# Luna Pérez Troncoso

#-----------LIBRARIES LOADING-------------


import streamlit as st
from streamlit import pdf
import pandas as pd
import numpy as np
import time
import matplotlib as mpl
import sklearn
import plotly
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as iPipeline
from imblearn.over_sampling import SMOTENC
from sklearn.feature_selection import SelectKBest,RFECV
from xgboost import XGBClassifier
import pickle

#-----------DATAFRAME AND MODEL LOADING-------------

train_df=pd.read_csv("./data/train.csv",index_col=0)
test_df=pd.read_csv("./data/test.csv",index_col=0)
processed=pd.read_csv("./data/processed.csv",index_col=0)
raw=pd.read_csv("./data/raw.csv",index_col=0)
with open("./models/stacking_model1", 'rb') as archivo_entrada:
    model1 = pickle.load(archivo_entrada)
with open("./models/stacking_model2", 'rb') as archivo_entrada:
    model2 = pickle.load(archivo_entrada)

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="Parkinson's Disease Model",
    page_icon="🧠",
    layout="wide"
)

# ---------- CUSTOM CSS STYLING ----------

st.markdown(
"""
    <style>
    /* Highlight analysis box */
    .highlight-box {
        border-left: 6px solid #0747d4;
        background-color: #F0F2F6;
        color: black;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14.5px;
    }

    </style>
    """
  ,unsafe_allow_html=True)


# ---------- APP PAGES FUNCTIONS ----------


def home_page():
    st.header("🏠 Home")
    
    st.markdown("""
   
Welcome to this web platform dedicated to presenting a comprehensive machine learning model developed to support the **predictive identification of Parkinson’s disease**. 
The primary objective of this site is to document the full analytical pipeline, from the raw data to the final predictive framework. 
In addition to providing methodological transparency, this website also offers an interactive component that enables users to **experiment with the predictive model themselves**, allowing them to explore how the system behaves under different input conditions.
The project leverages a set of demographic, lifestyle, clinical, cognitive, and symptom-related variables collected from a patient cohort. 
By integrating these diverse features into a unified structure, the study evaluates multiple machine learning algorithms and examines their capacity to accurately identify individuals at risk.

This website is organized into several sections to support a structured and transparent exploration of the project:

- **📚 Introduction**: An academic introduction outlining the motivation for applying machine learning to Parkinson’s disease prediction and the importance of early detection in clinical practice.
- **🗐 Data Description and Sources**: A detailed explanation of the dataset used in the project, including its origin, characteristics, and rationale for selection.
- **📂 Raw Data**: Direct access to the unprocessed dataset for transparency and reproducibility.
- **📊 Global Statistics**: A summary of descriptive statistics providing insights into the main distributions and demographic patterns observed within the data.
- **📈 Exploratory Data Analysis**: A comprehensive analytical section examining variable relationships, trends, and potential predictive indicators.
- **🎯 Interactive Predictions**:   A dedicated section in which users can directly interact with the predictive model, input custom values, and observe how the system generates risk estimates.
- **🙋🏻‍♀️ About the Author**: A brief professional profile of the project author, including background and scientific interests.    
     
This structure aims to ensure clarity, methodological rigor, and full reproducibility, offering a complete and transparent perspective on the development and evaluation of a machine learning–based predictive model for Parkinson’s disease.
    """)
    
def introduction_page():
    st.header("📚 Introduction")
    
    st.markdown("""
<h5 style="text-align: center;color: black;"> <b>What Is Parkinson’s Disease?</b></h5>
                  
Parkinson’s disease (PD) is a **progressive neurodegenerative disorder** that primarily affects the brain regions responsible for movement control. It occurs when the neurons that produce **dopamine**, a key neurotransmitter involved in coordinating smooth and balanced muscle activity, begin to deteriorate over time.

The condition is typically characterized by a combination of **motor symptoms**, such as tremor, muscle rigidity, bradykinesia (slowness of movement), and postural instability. Many individuals also experience **non-motor symptoms**, including sleep disturbances, cognitive changes, mood alterations, and autonomic dysfunction.

Although the exact cause of Parkinson’s disease remains unknown, research suggests a multifactorial origin involving **genetic**, **environmental**, and **age-related** factors. While there is currently no cure, early detection and comprehensive clinical management can significantly improve quality of life and slow symptom progression.

<h5 style="text-align: center;color: black;"> <b>Why Early Detection Is Crucial in Parkinson’s Disease?</b></h5>    
                   
The **early detection of PD** is a growing priority within both clinical practice and research, as the condition often progresses silently before clear symptoms emerge. Despite advances in medical imaging and neurological assessment, **many individuals remain undiagnosed until the disease has already progressed**, limiting the effectiveness of available therapeutic interventions. Consequently, **timely diagnosis** can significantly **influence patient outcomes and long-term quality of life**. Developing a **predictive model** for Parkinson’s disease therefore represents a crucial step toward identifying individuals at risk long before traditional diagnostic criteria are met.

<h5 style="text-align: center;color: black;"> <b>Why develop a machine learning–based predictive model?</b></h5>  
                     
A reliable predictive system has the potential to support clinicians in recognizing subtle signs that might otherwise go unnoticed. Early identification could allow for **timelier monitoring, lifestyle adjustments, and targeted therapeutic strategies** that may slow **disease progression or improve quality of life**. Additionally, predictive modeling can help researchers **gain deeper insight into the complex interactions that contribute to the onset of neurodegenerative disorders**.
   
Beyond clinical impact, creating a predictive model encourages the integration of modern **data-driven approaches** into neurological healthcare. As medicine increasingly embraces digital tools, predictive modeling stands out as a promising path to **more personalized and proactive patient care**. In the context of Parkinson’s disease, such a model could become an **essential component of future screening programs**, ultimately **contributing to earlier intervention**, improved outcomes, and a more informed understanding of this challenging condition.

<h5 style="text-align: center;color: black;"> <b>What is the objetive of this project?</b></h5>    
                   
A set of demographic, lifestyle, clinical, cognitive, and symptom-related variables collected from a patient cohort, was used in this project. By integrating these diverse variables into a unified predictive framework, the project seeks to **evaluate multiple machine learning algorithms** and determine their capability to accurately identify patients at risk. The goal is not only to achieve strong **predictive performance** but also to explore **feature importance** and interpretability, enabling clinicians and researchers to better understand which factors contribute most meaningfully to diagnostic outcomes. Ultimately, this work aims to support the development of practical, **data-driven tools** that may assist in earlier and more reliable detection of Parkinson’s disease.
    """,unsafe_allow_html=True)

def description_page():
    st.header("🗐 Data Description and Source")
    st.markdown("""
<h5 style="text-align: center;color: black;"> <b>Data Description</b></h5>        

The dataset includes a wide range of features that capture different dimensions of patient health: **Demographic variables, Lifestyle Factors, Medical History, Clinical Meassurements, Cognitive and Functional Assessments and Symptom indicators.** The features and categorical data encoding are the following:
   
**Patient ID**
- **PatientID:** A unique identifier assigned to each patient (3058 to 5162).

**Demographic Details**
- **Age:** The age of the patients ranges from 50 to 90 years.
- **Gender:** Gender of the patients  
  - 0: Male  
  - 1: Female
- **Ethnicity:**  
  - 0: Caucasian  
  - 1: African American  
  - 2: Asian  
  - 3: Other
- **EducationLevel:**  
  - 0: None  
  - 1: High School  
  - 2: Bachelor's  
  - 3: Higher

**Lifestyle Factors**
- **BMI:** Body Mass Index (15–40).
- **Smoking:**  
  - 0: No  
  - 1: Yes
- **AlcoholConsumption:** Weekly alcohol consumption in units (0–20).
- **PhysicalActivity:** Weekly physical activity in hours (0–10).
- **DietQuality:** Diet quality score (0–10).
- **SleepQuality:** Sleep quality score (4–10).

**Medical History**
- **FamilyHistoryParkinsons:**  
  - 0: No  
  - 1: Yes
- **TraumaticBrainInjury:**  
  - 0: No  
  - 1: Yes
- **Hypertension:**  
  - 0: No  
  - 1: Yes
- **Diabetes:**  
  - 0: No  
  - 1: Yes
- **Depression:**  
  - 0: No  
  - 1: Yes
- **Stroke:**  
  - 0: No  
  - 1: Yes

**Clinical Measurements**
- **SystolicBP:** 90–180 mmHg
- **DiastolicBP:** 60–120 mmHg
- **CholesterolTotal:** 150–300 mg/dL
- **CholesterolLDL:** 50–200 mg/dL
- **CholesterolHDL:** 20–100 mg/dL
- **CholesterolTriglycerides:** 50–400 mg/dL

**Cognitive and Functional Assessments**
- [**UPDRS:**](https://neurotoolkit.com/updrs/) 0–199 (higher = greater severity)
- [**MoCA:**](https://www.mdcalc.com/calc/10044/montreal-cognitive-assessment-moca) 0–30 (lower = cognitive impairment)
- **FunctionalAssessment:** 0–10 (lower = greater impairment)

**Symptoms**
- **Tremor:** 0 = No, 1 = Yes  
- **Rigidity:** 0 = No, 1 = Yes  
- **Bradykinesia:** 0 = No, 1 = Yes  
- **PosturalInstability:** 0 = No, 1 = Yes  
- **SpeechProblems:** 0 = No, 1 = Yes  
- **SleepDisorders:** 0 = No, 1 = Yes  
- **Constipation:** 0 = No, 1 = Yes  

**Diagnosis Information**
- **Diagnosis:** 0 = No Parkinson's Disease, 1 = Parkinson's Disease Diagnosed

**Confidential Information**
- **DoctorInCharge:** Confidential; value is "DrXXXConfid" for all patients.

---

<h5 style="text-align: center;color: black;"> <b>Data Source</b></h5>    
                  
As part of this project, I selected a [**synthetic dataset from Kaggle**](https://www.kaggle.com/datasets/rabieelkharoua/parkinsons-disease-dataset-analysis) generated by Mr. Rabie El Kharoua, to support the development of a predictive model for Parkinson’s disease. From a data science perspective, **this choice was driven by several important considerations:**

1. Working with **real patient data typically involves strict privacy regulations**, including HIPAA, GDPR, and institutional review requirements. These constraints make it difficult to share datasets publicly, reproduce analyses, or collaborate openly. Since one of my objectives was to create a **fully transparent and shareable workflow, using synthetic data ensured** that the entire project, including the dataset, code, and results, could be made **publicly available without violating confidentiality or ethical standards**.Ç

2. High-quality **synthetic datasets offer a practical alternative when real medical data is not accessible**. They are designed to mimic the statistical behavior and structure of real-world clinical information while removing any identifiable patient details. This makes them suitable for exploratory modeling, feature evaluation, and methodological experimentation. 
   
3. Using a Kaggle dataset allowed me to **create a resource that is open, reproducible, and widely accessible to other researchers and practitioners**. This supports the broader goal of making the project approachable to anyone who wants to understand, replicate, or extend the analysis.

Overall, the use of synthetic data enabled a balance between **ethical responsibility** and **scientific transparency**, allowing the project to remain publicly sharable while still demonstrating the workflow of developing a predictive model for Parkinson's disease.

    """,unsafe_allow_html=True)
    col1, col2, col3, = st.columns([6,4,6],gap="large",vertical_alignment="center")
    with col2:
        with st.container(border=True):
            st.image("./img/kaggle.png",use_container_width=True)
        


def raw_data_page():
    st.markdown(" In this page you can see the complete raw dataset that was employed for this proyect:**")
    st.header("📂 Raw Data")
    st.dataframe(raw)
    st.markdown('''*Note: you can download the raw data by clicking on download as csv icon ➜], for searching numbers in the dataframe click the search icon 🔍. Aditionally you can view the dataframe on full screen by clicking ⛶*''')
def statistics():
    st.header("📊 General Statistics")
    st.markdown(" In this page count, mean, standard deviation, minimun, maximum and quartiles **information of the different features of the train dataset** are shown:")
    st.dataframe(train_df.describe())
    st.markdown('''*Note: you can download the raw data by clicking on download as csv icon ➜], for searching numbers in the dataframe click the search icon 🔍. Aditionally you can view the dataframe on full screen by clicking ⛶*''')

def eda_page():
    st.header("📈 Exploratory Data Analysis")
    features=['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
       'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
       'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension',
       'Diabetes', 'Depression', 'Stroke', 'SystolicBP', 'DiastolicBP',
       'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
       'CholesterolTriglycerides', 'UPDRS', 'MoCA', 'FunctionalAssessment',
       'Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability',
       'SpeechProblems', 'SleepDisorders', 'Constipation', 'Diagnosis']
    graph=st.radio("Select graphical visualization",["Diagnosis (target) Balance", "Correlation Heatmap","Categorical Features Representation by Diagnosis","Numerical Features Scatter"],index=0)
    if graph=="Correlation Heatmap":
        fig = ff.create_annotated_heatmap(z=train_df.corr().values,x=features,y=features,showscale=True,annotation_text=train_df.corr().values.round(1),xgap=2,ygap=2,visible=True,colorscale=plotly.colors.diverging.Picnic,
                                                zmin=-1,zmax=1,colorbar=dict(title=dict(text="Pearson correlation coefficient",side="top")),font_colors=["black"])
        fig.update_layout(xaxis=dict(side="bottom"),title=dict(text="Correlation Heatmap",x=0.5,xanchor='center',font=dict(size=25)))
        fig.layout.height = 800
        st.plotly_chart(fig, use_container_width=True) 
        with st.expander(label="Show expanded correlations between features and target"):
            cmap = mpl.colormaps['bwr']
            col1, col2, col3 = st.columns(3)
            with col2:
                st.image("./img/colorbar.png",use_container_width=True)
                st.dataframe(train_df.corr().iloc[-1,0:-2].to_frame().sort_values(by="Diagnosis",ascending=False).style.background_gradient(cmap=cmap, axis=0,vmin=-1,vmax=1),height=1150)

    if graph=="Diagnosis (target) Balance":
        diagnosis_percentage=(train_df.Diagnosis.value_counts(normalize=True,ascending=True)*100)
        diagnosis_percentage.index=["No PD","PD"]
        diagnosis_percentage.name="Percentage (%)"
        fig = px.pie(diagnosis_percentage,values="Percentage (%)",names=["Parkinson's Disease Patients","Healthy Patients"], title="Percentage of Parkinson's Disease (PD) Patients and Healthy Patients",height=700)
        col1,col2,col3=st.columns([1,5,1])
        with col2:
            st.plotly_chart(fig, use_container_width=True) 
    if graph=="Categorical Features Representation by Diagnosis":
        cat_input=st.selectbox("Choose the categorical feature to study",['Smoking', 'Gender','EducationLevel','Ethnicity','Family History Parkinsons', 'Traumatic Brain Injury','Hypertension', 'Diabetes', 'Depression', 'Stroke','Tremor', 'Rigidity', 'Bradykinesia', 'Postural Instability','Speech Problems', 'Sleep Disorders', 'Constipation'])
        if cat_input == 'Family History Parkinsons':
            cat='FamilyHistoryParkinsons'
        elif cat_input == 'Traumatic Brain Injury':
            cat = 'TraumaticBrainInjury'
        elif cat_input == 'Postural Instability':
            cat = 'PosturalInstability'
        elif cat_input == 'Speech Problems':
            cat='SpeechProblems'
        elif cat_input== 'Sleep Disorders':
            cat='SleepDisorders'
        else:
            cat=cat_input
        if cat_input in ['Smoking', 'Family History Parkinsons', 'Traumatic Brain Injury','Hypertension', 'Diabetes', 'Depression', 'Stroke','Tremor', 'Rigidity', 'Bradykinesia', 'Postural Instability','Speech Problems', 'Sleep Disorders', 'Constipation','Gender']:
            df_plot_both=(train_df[train_df.Diagnosis==1][cat].value_counts(normalize=True).sort_index()*100).to_list()
            df_plot_both.extend((train_df[train_df.Diagnosis==0][cat].value_counts(normalize=True).sort_index()*100).to_list())
            df_plot=pd.DataFrame({"Diagnosis":["No PD","No PD","PD","PD"],"value":df_plot_both,"category":["No "+cat_input,cat_input,"No "+cat_input,cat_input]})
        elif cat=="Gender":
            df_plot_both=(train_df[train_df.Diagnosis==1]["Gender"].value_counts(normalize=True,sort=False).sort_index()*100).to_list()
            df_plot_both.extend((train_df[train_df.Diagnosis==0]["Gender"].value_counts(normalize=True,sort=False).sort_index()*100).to_list())
            df_plot=pd.DataFrame({"Diagnosis":["No PD","No PD","PD","PD"],"value":df_plot_both,"category":["Male","Female","Male","Female"]})
        elif cat=="Ethnicity":
            df_plot_both=(train_df[train_df.Diagnosis==1]['Ethnicity'].value_counts(normalize=True).sort_index()*100).to_list()
            df_plot_both.extend((train_df[train_df.Diagnosis==0]['Ethnicity'].value_counts(normalize=True).sort_index()*100).to_list())
            df_plot=pd.DataFrame({"Diagnosis":["No PD","No PD","No PD","No PD","PD","PD","PD","PD"],"value":df_plot_both,"category":["Caucasian", "African American","Asian","Other","Caucasian", "African American","Asian","Other"]})
        elif cat=="EducationLevel":
            df_plot_both=(train_df[train_df.Diagnosis==1]['EducationLevel'].value_counts(normalize=True,sort=False)*100).to_list()
            df_plot_both.extend((train_df[train_df.Diagnosis==0]['EducationLevel'].value_counts(normalize=True,sort=False)*100).to_list())
            df_plot=pd.DataFrame({"Diagnosis":["No PD","No PD","No PD","No PD","PD","PD","PD","PD"],"value":df_plot_both,"category":["None", "High School", "Bachelor's","Higher","None", "High School", "Bachelor's","Higher"]})
        fig=px.bar(df_plot,x="Diagnosis",y="value",color="category",title=f"Percentage of {cat_input} by Diagnosis",height=700)
        fig.update_layout(yaxis=dict(title=dict(text="Percentage (%)")))
        col1,col2,col3=st.columns([1,5,1])
        with col2:
            st.plotly_chart(fig)
    if graph=="Numerical Features Scatter":
        num_features=["Age","BMI","Alcohol Consumption","Physical Activity","Diet Quality","Sleep Quality","Cholesterol Total","Cholesterol LDL","Cholesterol HDL","Cholesterol Triglycerides","UPDRS","MoCA","Functional Assessment","Diagnosis"]
        x=st.pills("Select x axis", num_features, default="Diagnosis")
        if x != None:
            x_data=x.replace(" ","")
            num_features_y=num_features
            num_features_y.remove(x)
            y=st.pills("Select y axis", num_features_y)
            if y != None:
                y_data=y.replace(" ","")
                col1,col2,col3 =st.columns([1,5,1],gap="large")
                if x_data=="Diagnosis":
                    with col2:
                        fig = px.violin(train_df, x=x_data,y=y_data, box=True,height=700,title=f"{y} distribution by diagnosis violin plot")
                        fig.update_layout(xaxis=dict(title=dict(text=x)),yaxis=dict(title=dict(text=y)))
                        st.plotly_chart(fig)
                elif y_data=="Diagnosis":
                    with col2:
                        fig = px.violin(train_df, x_data, y_data,orientation="h", box=True,height=700,title=f"{y} distribution by diagnosis violin plot")
                        st.plotly_chart(fig)
                else:
                    with col2:
                        fig = px.scatter(train_df,x_data, y_data,height=700,title=f"{x} vs {y} Scatterplot ")
                        st.plotly_chart(fig,use_container_width=True)

def prediction_page():
    st.header("🎯 Interactive Predictions")
    help_message="**Accuracy score** meassure the **pertange of correct predictions** in comparison with all the predictions whereas **sensitibity score** describe per **percentage of patients with Parkinson's Disease detected**"
    st.markdown("")
    num_model=st.segmented_control("Choose the predictive model",["**Model 1** (accuracy oriented model)", "**Model 2** (sensitivity oriented model)"],help=help_message,default="**Model 1** (accuracy oriented model)")
    if num_model == "**Model 1** (accuracy oriented model)":
        selected_model=model1
    if num_model == "**Model 2** (sensitivity oriented model)":
        selected_model=model2
    st.markdown("Explore more details of the ML predictive model develpment and scores on this [**github repository**](https://github.com/LunaPerezT/Parkinson-s-Disease-Predictive-ML-Model)")
    GENDER_MAP = {"Male": 0, "Female": 1}
    ETHNICITY_MAP = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
    EDUCATION_MAP = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}
    BINARY_MAP = {"No": 0, "Yes": 1}
    st.markdown("---")
    st.markdown("Please fill out the following details to assess potential risk factors.")
    inputs = {}

    # --- Section 1: Demographic Details ---
    st.subheader("1. Demographic Details")

    # Age
    input_Age = st.slider("Age (years)", min_value=0, max_value=120, value=65, step=1)
    
    # Ethnicity
    col1,col2=st.columns(2)
    with col1:
        selected_ethnicity = st.selectbox("Ethnicity", list(ETHNICITY_MAP.keys()), index=0)
    input_Ethnicity = ETHNICITY_MAP[selected_ethnicity]

    # EducationLevel
    with col2:
        selected_education = st.selectbox("Education Level", list(EDUCATION_MAP.keys()), index=1)
    input_EducationLevel = EDUCATION_MAP[selected_education]
   
    # Gender
    selected_gender = st.radio("Gender", list(GENDER_MAP.keys()), index=0)
    input_Gender = GENDER_MAP[selected_gender]


    # --- Section 2: Lifestyle Factors ---
    st.subheader("2. Lifestyle Factors")

    # BMI
    input_BMI = st.slider("Body Mass Index (BMI)", min_value=15.0, max_value=40.0, value=25.0, step=0.1,help="Body Mass Index (BMI): [BMI Calculator](https://www.nhlbi.nih.gov/calculate-your-bmi)")
    
    # AlcoholConsumption
    input_AlcoholConsumption = st.slider("Weekly Alcohol Consumption (units)", min_value=0, max_value=30, value=0, step=1,help="Weekly alcohol consumption in units: [alcohol units calculator](https://alcoholchange.org.uk/alcohol-facts/interactive-tools/unit-calculator)")
    
    # PhysicalActivity
    input_PhysicalActivity = st.slider("Weekly Physical Activity (hours/week)", min_value=0.0, max_value=15.0, value=3.0, step=0.1,help="Weekly physical activity in hours per week.")
    
    # DietQuality
    input_DietQuality = st.slider("Diet Quality Score", min_value=0, max_value=10, value=5, step=1)

    # SleepQuality
    input_SleepQuality = st.slider("Sleep Quality Score", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

    # Smoking
    selected_smoking = st.radio("Smoking", list(BINARY_MAP.keys()), index=0, key="smoking",help="¿Is the patient currently a smoker?")
    input_Smoking = BINARY_MAP[selected_smoking]
    

    # --- Section 3: Medical History ---
    st.subheader("3. Medical History")
    
    # Helper function for binary radio buttons
    def create_binary_radio(label, key):
        selected = st.radio(label, list(BINARY_MAP.keys()), index=0, key=key)
        return BINARY_MAP[selected]
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            input_FamilyHistoryParkinsons = create_binary_radio("Family History of Parkinson's", "hist_parkinsons")
        with st.container(border=True):
            input_TraumaticBrainInjury = create_binary_radio("Traumatic Brain Injury (TBI)", "hist_tbi")
    with col2:
        with st.container(border=True):
            input_Hypertension = create_binary_radio("Hypertension", "hist_hypertension")
        with st.container(border=True):
            input_Diabetes = create_binary_radio("Diabetes", "hist_diabetes")
    with col3:
        with st.container(border=True):
            input_Depression = create_binary_radio("Depression", "hist_depression")
        with st.container(border=True):
            input_Stroke = create_binary_radio("Stroke", "hist_stroke")

    # --- Section 4: Clinical Measurements ---
    st.subheader("4. Clinical Measurements")

    col1, col2 = st.columns(2)
    
    with col1:
        input_SystolicBP = st.slider("Systolic BP (mmHg)", min_value=90, max_value=180, value=120, step=1)
        input_CholesterolTotal = st.slider("Total Cholesterol (mg/dL)",min_value=150, max_value=300, value=200, step=5)
        input_CholesterolLDL = st.slider("LDL Cholesterol (mg/dL)", min_value=50, max_value=200, value=100, step=5)
    with col2:
        input_DiastolicBP = st.slider("Diastolic BP (mmHg)", min_value=60, max_value=120, value=80, step=1,)
        input_CholesterolHDL = st.slider("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50, step=5)
        input_CholesterolTriglycerides = st.slider("Triglycerides (mg/dL)", min_value=50, max_value=400, value=150, step=10)

    # --- Section 5: Cognitive and Functional Assessments ---
    st.subheader("5. Cognitive and Functional Assessments")

    input_UPDRS = st.slider("UPDRS Score", min_value=0, max_value=199, value=50, step=1, help="Unified Parkinson's Disease Rating Scale (higher = greater severity): [UPDRS test webpage](https://neurotoolkit.com/updrs/)")
    input_MoCA = st.slider("MoCA Score", min_value=0, max_value=30, value=25, step=1,help="Montreal Cognitive Assessment (lower = cognitive impairment): [MoCA test webpage](https://www.mdcalc.com/calc/10044/montreal-cognitive-assessment-moca)") 
    input_FunctionalAssessment = st.slider("Functional Assessment Score", min_value=0.0, max_value=10.0, value=5.0, step=0.1, help="Score (lower = greater impairment).")

    # --- Section 6: Symptoms ---
    st.subheader("6. Current Symptoms")
    col1,col2,col3,col4=st.columns(4)
    with col1:
        with st.container(border=True):
            input_Tremor = create_binary_radio("Tremor", "symp_tremor")
    with col2:
        with st.container(border=True):
            input_Rigidity = create_binary_radio("Rigidity", "symp_rigidity")
    with col3:
        with st.container(border=True):
            input_SleepDisorders = create_binary_radio("Sleep Disorders", "symp_sleep")
    with col4:
        with st.container(border=True):
            input_Constipation = create_binary_radio("Constipation", "symp_constipation")
    col0,col1,col2,col3,col4 = st.columns([1,2,2,2,1])
    with col1:
        with st.container(border=True):
            input_Bradykinesia = create_binary_radio("Bradykinesia (Slowness of movement)", "symp_bradykinesia")
    with col2:
        with st.container(border=True):
            input_PosturalInstability = create_binary_radio("Postural Instability (Balance problems)", "symp_instability")
    with col3:
        with st.container(border=True):
            input_SpeechProblems = create_binary_radio("Speech Problems (poor phonation, articulation...)", "symp_speech")
 

    # --- Submit Button (Optional for demonstration) ---
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner('Calculating prediction ...'):
            input=[input_Age, input_Gender, input_Ethnicity, input_EducationLevel, input_BMI, input_Smoking, input_AlcoholConsumption, input_PhysicalActivity, input_DietQuality, input_SleepQuality,
            input_FamilyHistoryParkinsons, input_TraumaticBrainInjury, input_Hypertension, input_Diabetes, input_Depression, input_Stroke, input_SystolicBP, input_DiastolicBP,input_CholesterolTotal, 
            input_CholesterolLDL, input_CholesterolHDL, input_CholesterolTriglycerides, input_UPDRS, input_MoCA, input_FunctionalAssessment,input_Tremor, input_Rigidity, input_Bradykinesia, 
            input_PosturalInstability,input_SpeechProblems, input_SleepDisorders, input_Constipation]
            time.sleep(1) 
        st.subheader("Prediction Results")
        prediction=selected_model.predict([input])
        if prediction == 1:
            st.error("According to the submited data the patients is predicted to have **Parkinson's Disease** (Status **1**)")
            probability=selected_model.predict_proba([input])[0,1]
            st.metric(label="Estimated Probability (Status 1)", value=f"{np.round(probability*100)}%")
            st.warning("""
                **IMPORTANT NOTE:** The predictions are based on model trained with a synthetic dataset. **Results doesn't confirm a diagnosis of Parkinson's Disease**. 
                Nevertheless, based on the submited data **we strongly encourage the patient to visit a specialized medical professional** 
            """)
        else:
            st.success("According to the submited data the patients is predicted to be **Healthy** (Status **0**)")
            probability=selected_model.predict_proba([input])[0,0]
            st.metric(label="Estimated Probability (Status 0)", value=f"{np.round(probability*100)}%")
            st.info(""" The model sugest a negative results. Anyways the predictions are based on model trained with a synthetic dataset. Consenquenty, **these results should not replace specialized medical attention**""")
    st.markdown("---")
    with st.expander("Upload your own dataset in csv format",expanded=False):
        st.markdown("In this section you predict the diagnosis of multiple patients with uploading your own dataset in csv format")
        st.markdown("")
        st.warning("The imported data must have the following columns names and order: Age, Gender, Ethnicity, EducationLevel, BMI, Smoking,AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality,FamilyHistoryParkinsons, TraumaticBrainInjury, Hypertension,Diabetes, Depression, Stroke, SystolicBP, DiastolicBP,CholesterolTotal, CholesterolLDL, CholesterolHDL,CholesterolTriglycerides, UPDRS, MoCA, FunctionalAssessment,Tremor, Rigidity, Bradykinesia, PosturalInstability,SpeechProblems, SleepDisorders, Constipation,")
        uploaded_file = st.file_uploader("",accept_multiple_files=False, type="csv")
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            num_input_model=st.segmented_control("Choose the predictive model:",["**Model 1** (accuracy oriented)", "**Model 2** (sensitivity oriented)"],help=help_message,key="input_ML")
            if num_input_model == "**Model 1** (accuracy oriented)":
                input_model=model1
            if num_input_model == "**Model 2** (sensitivity oriented)":
                input_model=model2 
            if num_input_model is not None:
                if st.button("Predict CSV", type="primary", use_container_width=True):
                    try:
                        predictions=input_model.predict(input_df)
                        st.markdown("""Predictions Map:   
                                    
                                         0 --> Healthy Patient 

                                         1 --> Parkison's Disease Patient""")
                        col1,col2,col3=st.columns([8,9,9],gap="large")
                        with col2:
                            st.dataframe(predictions)
                        st.markdown('''*Note: you can download the raw data by clicking on download as csv icon ➜], for searching numbers in the dataframe click the search icon 🔍. Aditionally you can view the dataframe on full screen by clicking ⛶*''')
                        st.info(""" The predictions are based on model trained with a synthetic dataset. Consenquenty, **these results should not replace specialized medical attention**""")
                    except:
                        st.error("Uploaded data does not follow the described format")

def author_page():
    st.header("🙋🏻‍♀️ About the Author")
    st.markdown('''
    I’m **Luna Pérez Troncoso**, a **Data Scientist** with a strong foundation in **Artificial Intelligence, data analytics, and computational modeling**, originally shaped through my background in **science and research**. Over time, I’ve transitioned my **analytical and experimental mindset** into the tech field, where I design **data-driven solutions** that **transform complex information into actionable insights**.
         
    I have experience working across the **full data pipeline**, from data acquisition, cleaning, and exploration to building, validating, and deploying predictive models. My focus is on **leveraging machine learning and statistical methods to uncover patterns, optimize processes, and support strategic decisions**.
           
    What defines my approach is a balance between **technical precision and creativity**. I’m passionate about **connecting raw data with real-world** impact, collaborating with **cross-functional teams**, and communicating insights in a way that drives innovation.
    ''')
    col1, col2= st.columns(2,gap="large",vertical_alignment="center")
    with col1:
        with st.container(border=True):
            col1_1, col1_2 = st.columns(2,vertical_alignment="center")
            with col1_1:
                st.image("./img/linkedin.jpg",use_container_width=True)
            with col1_2:
                st.markdown('''
                <center>
                                
                Visit my [linkedIn profile](https://www.linkedin.com/in/luna-p%C3%A9rez-troncoso-0ab21929b/)
                            
                </center>
                ''',unsafe_allow_html=True)
    with col2:
        with st.container(border=True):
            col2_1, col2_2 = st.columns(2,vertical_alignment="center")
            with col2_1:
                st.image("./img/github.png",use_container_width=True)
            with col2_2:
                st.markdown('''
                <center>
                                
                Explore my projects on my [Github profile](https://github.com/LunaPerezT)
                            
                </center>
                ''',unsafe_allow_html=True)


# ---------- APP HEADER ----------

st.markdown('''<h1 style="text-align: center;color: black;"> <b>Parkinson's Disease Predictive Machine Learning Model</b></h1>
    <h5 style="text-align: center;color: gray"> Luna Pérez Troncoso </h5>''', unsafe_allow_html=True)
st.markdown("---")

# ---------- SIDEBAR NAVIGATION ----------

st.sidebar.title("Navigation Menu")
selection = st.sidebar.radio("Select Section:",
        ("🏠 Home",
        "📚 Introduction",
        "🗐 Data Description and Source",
        "📂 Raw Data",
        "📊 General Statistics",
        "📈 Exploratory Data Analysis",
        "🎯 Interactive Predictions",
        "🙋🏻‍♀️ About the Author"),index=0)
   
# ---------- APP BODY ----------
if selection == "🏠 Home":
    home_page()
elif selection =="📚 Introduction":
    introduction_page()
elif selection =="🗐 Data Description and Source":
    description_page()
elif selection == "📂 Raw Data":
    raw_data_page()
elif selection =="📊 General Statistics":
    statistics()
elif selection == "📈 Exploratory Data Analysis":
    eda_page()
elif selection == "🎯 Interactive Predictions":
    prediction_page()
elif selection == "🙋🏻‍♀️ About the Author":   
    author_page()
    
# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 1em;'>© 2025 Parkinson's Disease Predictive Machine Learning Model EDA — Luna Pérez Troncoso </p>",unsafe_allow_html=True)
