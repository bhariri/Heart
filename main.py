# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:39:04 2022

@author: Basman Hariri
""" 


import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as gos
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="heart",
    layout="centered"
)



###################################################################################################################################################################
# Creating the Side Bar Navigation Panel
navigate = st.sidebar.radio('Navigation Side Bar',
                 ('Home Page', 'Summary Statistics', 'Pre ML Analysis',
                  'Model Evaluation','User App'))
imgside=Image.open('sidebar.jpg')
st.sidebar.image(imgside, use_column_width=True)

# Updating the Datset if needed
uploaded_file = st.file_uploader("Upload updated dataset")
if uploaded_file is None:
    df = pd.read_csv("heart.csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if st.button('Show a sample of the data'):
        st.write(df.head())
#####################################################################################################################################################################
df.head()        
df.describe()

#checking for null values
df.isna().sum()

#features info
df.info()


# Creating the Home Page

if navigate == 'Home Page':
    # adding title
    title_col = st.columns(1)
    title_col[0].title("Heart Failure Prediction")
    
    
    # adding the home page image
    img=Image.open('image.jpg')
    st.image(img)

    
    # dashboard description
    st.header("Context")
    st.markdown("""Heart failure is a common event caused by cardiovascular diseases. Cardiovascular diseases (CVDs) are the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of Five CVD deaths are due to heart failures, and one-third of these deaths occur prematurely in people under 70 years of age. 
    People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
    """)
    # dataset info
    st.header("Dataset Information")
    st.markdown("""The dataset contains 11 features that can be used to predict a possible heart disease. It includes 12 variables describing 918 observations.
                Data is collected and combined from Five heart datasets with 11 common features which makes this dataset large and reach enough for research and educational purposes.""")


# Creating the Dashboard Page

if navigate == 'Summary Statistics':
    # adding an aligned title without using CSS
    st.markdown("<h1 style='text-align: center; color: blue;'>Exploratory Analysis of the Population</h1>", unsafe_allow_html=True)
    
    fig1=plt.figure(figsize=(12,20))
    plt.subplot(4,3,1)
    bins = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,100]
    sns.histplot(data=df, x="Age", kde=True, hue="HeartDisease", bins=bins);
    # sns.distplot(df['Age'],color='navy')
    plt.title("HD by Age",fontweight="bold")
    plt.subplot(4,3,2)
    sns.countplot(data=df,x="Sex", hue="HeartDisease")
    sns.despine(top=True,right=True)
    # plt.pie(df['Sex'].value_counts(),labels=['Male','Female'],autopct="%.1f%%")
    plt.title("HD by Sex",fontweight="bold")
    plt.subplot(4,3,3)
    # sns.countplot(df['ChestPainType'])
    sns.countplot(data=df,x="ChestPainType",hue="HeartDisease")
    plt.title("HD by Chest Pain Type",fontweight="bold")
    plt.subplot(4,3,4)
    # sns.distplot(df['RestingBP'],color='navy')
    bin=[75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,180,190,200]
    sns.histplot(data=df,x="RestingBP",hue="HeartDisease",bins=bin)
    plt.title("HD by RBP",fontweight="bold")
    plt.subplot(4,3,5)
    # sns.distplot(df['Cholesterol'],color='navy')
    bin=[100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445,450,455]
    sns.histplot(data=df,x="Cholesterol",hue="HeartDisease",bins=bin)
    plt.title("HD by Cholesterol Level",fontweight="bold")
    plt.subplot(4,3,6)
    # plt.pie(df['FastingBS'].value_counts(),labels=['under 120mg/dl','over 120mg/dl'],colors = ['tab:green', 'tab:red'],autopct="%.1f%%")
    sns.countplot(data=df,x="FastingBS", hue="HeartDisease")
    sns.despine(top=True,right=True)
    plt.title("HD by FBS 1>120mn\dl",fontweight="bold")
    plt.subplot(4,3,7)
    # sns.countplot(df['RestingECG'],palette=['#008000',"#FFA500","#FF0000"])
    sns.countplot(data=df,x="RestingECG",hue="HeartDisease")
    plt.title("HD by Resting Electrocardiogram Results ",fontweight="bold")
    plt.subplot(4,3,8)
    # sns.distplot(df['MaxHR'],color='navy')
    sns.histplot(data=df,x="MaxHR",hue="HeartDisease")
    plt.title("HD by Max Heart Rate",fontweight="bold")
    plt.subplot(4,3,9)
    # plt.pie(df['ExerciseAngina'].value_counts(),labels=['No','Yes'],autopct="%.1f%%")
    sns.countplot(data=df,x="ExerciseAngina", hue="HeartDisease")
    plt.title("HD by Exercise Induced-Angina",fontweight="bold")
    plt.subplot(4,3,10)
    # sns.distplot(df['Oldpeak'],color='navy')
    sns.histplot(data=df,x="Oldpeak",hue="HeartDisease")
    plt.title("ST depression induced by exercise/rest",fontweight="bold")
    plt.subplot(4,3,11)
    # plt.pie(df['ST_Slope'].value_counts(),labels=['Flat','Up','Down'],autopct="%.1f%%")
    sns.countplot(data=df,x="ST_Slope",hue="HeartDisease")
    plt.title("HD by ST Exercise Slope",fontweight="bold")
    plt.subplot(4,3,12)
    plt.pie(df['HeartDisease'].value_counts(),labels=["Yes","No"],colors = ['tab:red', 'tab:green'],autopct="%.1f%%")
    plt.title("Percentage of Heart Disease",fontweight="bold")

    st.pyplot(fig1)
########################################################################################################################################################################

# Creating the Data Analysis page

if navigate == 'Pre ML Analysis':
    
    data = df.copy()
    data['Sex'].replace('F', 0,inplace=True)
    data['Sex'].replace('M', 1,inplace=True)
    data['ChestPainType'].replace('TA', 1,inplace=True)
    data['ChestPainType'].replace('ATA', 2,inplace=True)
    data['ChestPainType'].replace('NAP', 3,inplace=True)
    data['ChestPainType'].replace('ASY', 4,inplace=True)
    data['RestingECG'].replace('LVH', 1,inplace=True)
    data['RestingECG'].replace('Normal', 2,inplace=True)
    data['RestingECG'].replace('ST', 3,inplace=True)
    data['ExerciseAngina'].replace('Y', 1,inplace=True)
    data['ExerciseAngina'].replace('N', 2,inplace=True)
    data['ST_Slope'].replace('Down', 1,inplace=True)
    data['ST_Slope'].replace('Flat', 2,inplace=True)
    data['ST_Slope'].replace('Up', 3,inplace=True)
    X = data.iloc[:,0:11]  #independent columns
    y = data.iloc[:,-1]    #target column 
    model = ExtraTreesClassifier()
    model.fit(X,y)
    
    #Feature importance
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
 
    
    #Feature importance 
                   
    fig2=feat_importances.nlargest(13).plot(kind='barh')
    plt.title("\n"
              "Feature Importance\n",fontweight="bold")
    st.pyplot(fig2.figure)
    st.markdown("\n"
            "Using the Extra Tree Classifier to extract the top features for the dataset helped to identify the significance of each feature.\n"
            "As per the above Plot, 'Chest Pain Type, 'Exercise Angina' and 'St Slope' are the top three significant features in our dataset.\n"
            " ")
    
    fig3=sns.heatmap(data.corr().replace(1.0, np.nan),annot=True,cmap=sns.diverging_palette(10, 10, n=9),fmt='.2f')
    plt.title("\n"
              "Correlation Matrix with Heatmap\n",fontweight="bold")
    st.pyplot(fig3.figure)
    st.markdown("\n"
            "The above 'seanborn library' Heatmap displays how much the variables are related between each other and to the target variable .\n"
            "From this heatmap we can observe that the ‘Chest Pain Type’ is highly related to the target variable. Compared to other relations we can say that chest pain contributes the most in prediction of presences of a heart disease \n"
            " ")    
    
    
# Creating the Model page

if navigate == 'Model Evaluation':
    #Logistic Regression
    data2=df.copy()
    data2['Sex'].replace('F', 0,inplace=True)
    data2['Sex'].replace('M', 1,inplace=True)
    data2['ChestPainType'].replace('TA', 1,inplace=True)
    data2['ChestPainType'].replace('ATA', 2,inplace=True)
    data2['ChestPainType'].replace('NAP', 3,inplace=True)
    data2['ChestPainType'].replace('ASY', 4,inplace=True)
    data2['RestingECG'].replace('LVH', 1,inplace=True)
    data2['RestingECG'].replace('Normal', 2,inplace=True)
    data2['RestingECG'].replace('ST', 3,inplace=True)
    data2['ExerciseAngina'].replace('Y', 1,inplace=True)
    data2['ExerciseAngina'].replace('N', 2,inplace=True)
    data2['ST_Slope'].replace('Down', 1,inplace=True)
    data2['ST_Slope'].replace('Flat', 2,inplace=True)
    data2['ST_Slope'].replace('Up', 3,inplace=True)
    X = data2.drop('HeartDisease', axis = 1)
    y = data2['HeartDisease']
    
    #Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Data Normalization
    X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
    X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values
    
    #Fitting the Model
    logre = LogisticRegression()
    logre.fit(X_train,y_train)

    #Prediction
    y_pred = logre.predict(X_test)
    actual = []
    predcition = []
    for i,j in zip(y_test,y_pred):
        actual.append(i)
        predcition.append(j)
    dic = {'Actual':actual,'Prediction':predcition}
        
    result  = pd.DataFrame(dic)
    
    
    st.header('Logistic regression Prediction')
    # st.plotly_chart(fig, use_container_width=True)
    
    print(accuracy_score(y_test,y_pred))
    st.code("Accuracy Level = 0.842391304347826")
    
    print(classification_report(y_test,y_pred))
    st.markdown("classification report")
    st.text("              precision    recall  f1-score   support  \n\n"

               "0       0.82      0.81      0.81        77  \n"
               "1       0.86      0.87      0.87       107  \n\n"

    "    accuracy                           0.84       184 \n"
    "   macro avg       0.84      0.84      0.84       184 \n"
    "weighted avg       0.84      0.84      0.84       184")
    
    st.markdown("\n"
                "The classification report of the model shows that 86% prediction of presence of heart disease were predicted correct and 82% of absence of heart disease were predicted correct, with a recall of 87% of actual positive and 81% of actual negative")
    #Confusion Matrix
    
    print(confusion_matrix(y_test,y_pred))
    
    
    fig4=sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='Blues')
    plt.title("Confusion Matrix\n",fontweight="bold")
    st.markdown(" \n"
                 " ")
    st.pyplot(fig4.figure)
    st.markdown("\n"
                "The Confusion Matrix True Positive value is 62 and true Negative was 93. And the False Positive came out to be 15 and False Negative is 14.")
    #Coefficents
    print(logre.intercept_)
    plt.figure(figsize=(10,12))
    coeffecients = pd.DataFrame(logre.coef_.ravel(),X.columns)
    coeffecients.columns = ['Coeffecient']
    coeffecients.sort_values(by=['Coeffecient'],inplace=True,ascending=False)
    #coeffecients
    plt.figure(figsize=(10,12))
    coeffecients = pd.DataFrame(logre.coef_.ravel(),X.columns)
    coeffecients.columns = ['Coeffecient']
    coeffecients.sort_values(by=['Coeffecient'],inplace=True,ascending=False)
    fig5=sns.heatmap(coeffecients,annot=True,fmt='.2f',cmap='Set2',linewidths=0.5)
    plt.title("Heat Map for Features Contribution\n",fontweight="bold")
    
    st.pyplot(fig5.figure)
    st.markdown(" \n"
                 "The important features contributing to the accuracy of the prediction are shown through the Heatmap in descending order. In silver color code, the most contributing features are 'the chest pain type', 'Old_Peak' and 'sex'.")
# Creating the User App page

if navigate == 'User App':
#same previous code again for this tab
#Logistic Regression
    data3=df.copy()
    data3['Sex'].replace('F', 0,inplace=True)
    data3['Sex'].replace('M', 1,inplace=True)
    data3['ChestPainType'].replace('TA', 1,inplace=True)
    data3['ChestPainType'].replace('ATA', 2,inplace=True)
    data3['ChestPainType'].replace('NAP', 3,inplace=True)
    data3['ChestPainType'].replace('ASY', 4,inplace=True)
    data3['RestingECG'].replace('LVH', 1,inplace=True)
    data3['RestingECG'].replace('Normal', 2,inplace=True)
    data3['RestingECG'].replace('ST', 3,inplace=True)
    data3['ExerciseAngina'].replace('Y', 1,inplace=True)
    data3['ExerciseAngina'].replace('N', 2,inplace=True)
    data3['ST_Slope'].replace('Down', 1,inplace=True)
    data3['ST_Slope'].replace('Flat', 2,inplace=True)
    data3['ST_Slope'].replace('Up', 3,inplace=True)
    X = data3.drop('HeartDisease', axis = 1)
    y = data3['HeartDisease']

#Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#     Data Normalization
#     X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
#     X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

#Scale the Feature Data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(X_train)
    test_features = scaler.transform(X_test)

#Fitting the Model
    logre = LogisticRegression()
    logre.fit(X_train,y_train)
    

#Prediction
    y_pred = logre.predict(X_test)
    actual = []
    predcition = []
    for i,j in zip(y_test,y_pred):
        actual.append(i)
        predcition.append(j)
        dic = {'Actual':actual,'Prediction':predcition}
    
    result  = pd.DataFrame(dic)

    fig = gos.Figure()


    fig.add_trace(gos.Scatter(x=np.arange(0,len(y_test)), y=y_test,
                mode='markers+lines',
                name='Test'))

    fig.add_trace(gos.Scatter(x=np.arange(0,len(y_test)), y=y_pred,
                mode='markers',
                name='Pred'))





    
# User Data Entry
    
    Age = st.slider('Age',min_value =1,max_value=100,step=1)
    sex = st.selectbox("Sex",options=['Male' , 'Female'])
    ChestPainType = st.selectbox("Chest Pain Type",options=['Typical Angia' , 'Atypical Angina','Non Anginal Pain','Asymptomatic'])
    RestingBP= st.slider('Resting BP',min_value=20,max_value=260,step =1)
    Cholesterol= st.slider('Cholesterol',min_value=85,max_value=605,step =1)
    FastingBS = st.selectbox("Diabetes ",options=['Yes' , 'No'])
    RestingECG = st.selectbox("Resting ECG",options=['LVH' , 'Normal','ST'])
    MaxHR= st.slider('Max HR',min_value=50,max_value=210,step =1)
    ExerciseAngina = st.selectbox("Exercise Anginal Pain ",options=['Yes' , 'No'])
    Oldpeak=st.slider('Oldpeak',min_value=-2,max_value=7,step =1)
    ST_Slope = st.selectbox("ST Slope",options=['Down' , 'Flat','Up'])
    
    sex = 1 if sex == 'Male' else 0
    
    if ChestPainType =='Typical Angia':
        ChestPainType = 1
    elif ChestPainType == 'Atypical Angina':
        ChestPainType = 2
    elif ChestPainType == 'Non Anginal Pain':
        ChestPainType = 3
    else:
        ChestPainType = 4
        
    if RestingECG =='LVH':
        RestingECG = 1
    elif RestingECG == 'Normal':
        RestingECG = 2
    else:
        RestingECG = 3
    
    FastingBS = 1 if FastingBS == 'Yes' else 0
    
    ExerciseAngina = 1 if ExerciseAngina == 'Yes' else 2

    if ST_Slope =='Down':
        ST_Slope = 1
    elif ST_Slope == 'Flat':
        ST_Slope = 2
    else:
        ST_Slope = 3
        
    user_data = scaler.transform([[Age , sex, ChestPainType , RestingBP, Cholesterol, FastingBS,RestingECG ,MaxHR ,ExerciseAngina , Oldpeak, ST_Slope]])
    prediction = logre.predict(user_data)
    predict_probability = logre.predict_proba(user_data)
    predict_probability            
    
    st.header('Data Provided by User')
    st.table(user_data)
    # prediction = logre.predict(user_data)
    # st.write(testpred)  
    # st.subheader(str(prediction))
    # st.subheader(str(predict_probability))
    
    if prediction [0] == 1:
        st.subheader('Patient is at Risk of Heart Failure')
    else:
        st.subheader('Patient is not at Risk of Heart Failure')
