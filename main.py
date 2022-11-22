import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import plotly.express as px




@st.cache
def loadData():
    df = pd.read_csv("/brain_stroke 3.csv")
    
    df=df.sample(frac=1).reset_index(drop=True)
    df2=df
    df['gender'] = pd.Categorical(df['gender']).codes
    df['ever_married'] = pd.Categorical(df['ever_married']).codes
    df['work_type'] = pd.Categorical(df['work_type']).codes
    df['Residence_type'] = pd.Categorical(df['Residence_type']).codes
    df['smoking_status'] = pd.Categorical(df['smoking_status']).codes
    return df , df2


# Basic preprocessing.
def preprocessing(df):
    # Assign X and y
    X = df.iloc[:, 0:10].values
    y = df.iloc[:, -1].values

    # y  Categorical data 
    le = LabelEncoder()
    y = le.fit_transform(y.flatten())

    # 1. Splitting X,y into Train & Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test, le

def  gender_encoding(str):
    if ('Female' in str ): 
      return 0 
    else : 
      return 1


def  stroke_decoding(pred):
    if (pred == 0 ): 
      original_title = '<p style="font-family:Courier; color:Green; font-size: 20px;">There is no risk of brain stroke</p>'
      st.markdown(original_title, unsafe_allow_html=True)
      #return 'There is no risk of brain stroke'
    else : 
      original_title = '<p style="font-family:Courier; color:Red; font-size: 20px;">There is a risk of brain stroke</p>'
      st.markdown(original_title, unsafe_allow_html=True)
      #return 'There is a risk of brain stroke'

def  yes_no_encoding(str):
    if ('No' in str ): 
      return 0 
    else : 
      return 1

def  residence_type_encoding(str):
    if ('Rural' in str ): 
      return 0 
    else : 
      return 1


def  work_type_encoding(str):
    if 'children' in str:
       return 0
    elif 'Govt_job' in str:
       return 1
    elif 'Private' in str:
       return 2
    elif 'Self-employed' in str:
       return 3


def  smoking_status_encoding(str):
    if 'Unknown' in str:
       return 0
    elif 'formerly smoked' in str:
       return 1
    elif 'never smoked' in str:
       return 2
    elif 'smokes' in str :
       return 3
       




# Training Decission Tree for Classification
#@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train, y_test):
    # Train the model
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)

    return score, report, tree, disp


# Training Neural Network for Classification.
#@st.cache(suppress_st_warning=True)
def neuralNet(X_train, X_test, y_train, y_test):
    # Scalling the data before feeding it to the Neural Network.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Instantiate the Classifier and fit the model.
    clf = MLPClassifier(solver='adam',activation='relu', alpha=1e-2, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score1 = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)

    return score1, report, clf, disp


# Training KNN Classifier
#@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test, k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)

    return score, report, clf, disp


# Training NaiveBayes Classifier
#@st.cache(suppress_st_warning=True)
def NaiveBayes_Classifier(X_train, X_test, y_train, y_test):
    # Create a Gaussian Classifier

    gnb = GaussianNB()
    # Train the model using the training sets
    gnb.fit(X_train, y_train)
    # Predict the response for test dataset

    y_pred = gnb.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100

    report = classification_report(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)

    return score, report, gnb, disp







# Accepting user data for predicting its Member Type
def accept_user_data(df):
    gender = st.radio("What's your gender",('Male', 'Female'))
    age = st.number_input('Enter your age',value=0,step=1)
    hypertension = st.radio("Do you have hypertension ?",('Yes', 'No'))
    heart_disease = st.radio("Do you have any kind of heart disease ?",('Yes', 'No'))
    ever_married = st.radio("Do you have ever get married ?",('Yes', 'No'))
    work_type = st.selectbox('Select your work type?',('Private', 'Self-employed', 'Govt_job','children'))
    Residence_type = st.selectbox('Select your residence type type?',('Rural', 'Urban'))
    avg_glucose_level = st.number_input('Enter your avg  glucose level',value=0.00,step=0.01)
    bmi = st.number_input('Enter your bmi',value=0.0,step=0.1)
    smoking_status = st.selectbox('What is your smoking status ?',('Unknown', 'never smoked',  'formerly smoked','smokes'))
    
    #smoking_status_code = df["smoking_status"].cat.categories.get_loc(smoking_status)

    user_prediction_data = np.array([gender_encoding(gender),age,yes_no_encoding(hypertension),yes_no_encoding(heart_disease),yes_no_encoding(ever_married),work_type_encoding(work_type),residence_type_encoding(Residence_type),avg_glucose_level,bmi,smoking_status_encoding(smoking_status)]).reshape(1, -1)
    return user_prediction_data










def main():
    st.title(
        "Mini Projet Analyse et fouille de données!")
    data,data2 = loadData()
    X_train, X_test, y_train, y_test, le = preprocessing(data)

    # Insert Check-Box to show the snippet of the data.
    if st.checkbox('Voir dataset'):
        st.subheader("Dataset  -->>>")
        st.write(data2.head())
        st.write(data.head())

    # ML Section
    choose_model = st.sidebar.selectbox("Choisissez le modèle",
                                        ["NONE", "Decision Tree", "Neural Network", "K-Nearest Neighbours",
                                         "Naive Bayes"])

    if (choose_model == "Decision Tree"):
        score, report, tree, disp = decisionTree(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Decision Tree model is: ")
        st.write(score, "%")
        st.text("Report of Decision Tree model is: ")
        # st.write(report)

        st.text("." + report)
        disp.plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())

        try:
            if (st.checkbox(
                    "Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data(data2)
                st.write(user_prediction_data)
                pred = tree.predict(user_prediction_data)
                stroke_decoding(pred)   # Inverse transform to get the original dependent value.
        except:
            pass

    elif (choose_model == "Neural Network"):
        score, report, clf, disp = neuralNet(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Neural Network model is: ")
        st.write(score, "%")
        st.text("Report of Neural Network model is: ")
        # st.write(report)

        st.text("." + report)
        disp.plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())

        try:
            if (st.checkbox(
                    "Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data(data2)
                scaler = StandardScaler()
                scaler.fit(X_train)
                user_prediction_data = scaler.transform(user_prediction_data)
                pred = clf.predict(user_prediction_data)
                #st.write("The Predicted Class is: ",
                 #      le.inverse_transform(pred)) 
                stroke_decoding(pred)
        except:
            pass

    elif (choose_model == "K-Nearest Neighbours"):

        n_neighbors = st.number_input('Choose k', value=1,min_value=1 , step=2)
        st.text("k :  ")
        st.write(n_neighbors)
        score, report, clf , disp = Knn_Classifier(X_train, X_test, y_train, y_test, n_neighbors)
        st.text("Accuracy of K-Nearest Neighbour model is: ")
        st.write(score, "%")
        st.text("Report of K-Nearest Neighbour model is: ")
        # st.write(report)

        st.text("." + report)
        disp.plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())

        try:
            if (st.checkbox(
                    "Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data(data2)
                pred = clf.predict(user_prediction_data)
                stroke_decoding(pred)   # Inverse transform to get the original dependent value.
        except:
            pass
    elif (choose_model == "Naive Bayes"):
        score, report, gnb, disp = NaiveBayes_Classifier(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Naive Bayes model is: ")
        st.write(score, "%")
        st.text("Report of Naive Bayes model is: ")
        # st.write(report)

        st.text("." + report)
        disp.plot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(plt.show())

        try:
            if (st.checkbox(
                    "Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data(data2)
                pred = gnb.predict(user_prediction_data)
                stroke_decoding(pred)  # Inverse transform to get the original dependent value.
        except:
            pass

    choose_viz = st.sidebar.selectbox("Choose the Visualization",["NONE","Total number of person married with the risk of brain stroke","Total number of person that have a heart diseases with the risk of brain stroke","Countribution of smoking status"])   
    
    if(choose_viz == "Total number of person married with the risk of brain stroke") :
        fig = px.histogram(data[data.stroke==1]['ever_married'], x ='ever_married')
        st.plotly_chart(fig)
    elif(choose_viz == "Total number of person that have a heart diseases with the risk of brain stroke") :
        fig = px.histogram(data[data.stroke==1]['heart_disease'], x ='heart_disease')
        st.plotly_chart(fig)
    elif(choose_viz == "Countribution of smoking status"):
        plot = data[data.stroke==1]['smoking_status'].plot.pie(y='mass', figsize=(5, 5))
        data[data.stroke==1].groupby(['smoking_status']).sum().plot(kind='pie')
        #fig = px.histogram(data['Member type'], x ='Member type')
        fig, ax = plt.subplots()
        ax = data[data.stroke==1].groupby(['smoking_status']).sum().plot(kind='pie')
        #ax.pie(sorted(data[data.stroke==1]['smoking_status'].values) )
        ax.axis('equal') 
        st.pyplot(fig)

# plt.hist(data['Member type'], bins=5)
# st.pyplot()

if __name__ == "__main__":
    main()
