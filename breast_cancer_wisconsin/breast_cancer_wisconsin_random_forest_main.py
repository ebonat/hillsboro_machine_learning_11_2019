

# Breast Cancer Diagnosis with Neural Networks -Keras 4
# https://www.youtube.com/watch?v=QiLHwCkx-YQ

# Implementing a Binary Classifier in Python
# https://medium.com/maheshkkumar/implementing-a-binary-classifier-in-python-b69d08d8da21

# How to handle Imbalanced Classification Problems in machine learning?
# https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/

import os
import sys
import time
import numpy as np                           
import pandas as pd                            
import matplotlib.pyplot as plt        
import matplotlib.ticker as ticker       
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier
import config
import pickle

import warnings                                  
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, classes, normalized=True, cmap='coolwarm'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.title("Classification Confusion Matrix")
    plt.show()
    
def numpy_unique_value_matrix(numpy_array):    
    try:
        unique, counts = np.unique(numpy_array, return_counts=True)
        unique_counts = np.asarray((unique, counts)).T        
    except:
        exception_message = sys.exc_info()[0]
        print("An error occurred. {}".format(exception_message))
    return unique_counts

def main():
#     LOAD
    csv_directory_file_input = os.path.join(os.path.dirname(__file__), "csv")            
    csv_path_file = config.BREAST_CANCER_WISCONSIN_CLEANED_TRAIN_CSV
    csv_directory_path_file = os.path.join(csv_directory_file_input, csv_path_file)    
    df_breast = pd.read_csv(filepath_or_buffer=csv_directory_path_file)
    
#     SELECT LABEL (TARGET)
    y = df_breast[config.BREAST_CANCER_WISCONSIN_LABEL]
    y_counts = y.value_counts()
    print("class count")
    print(y_counts)
    print()
    
#     y_unique_class = list(y.unique())
    y_unique_class = list(y.sort_values().unique())
    print("unique class")
    print(y_unique_class)
    print()

#     CONVERT Y SERIES TO NUMPY ARRAY
    y = np.array(y)      

#     NEED TO CHECK AND ANALYZE IMBALANCED DATA CLASS FOR Y?
    
#     SELECT FEATURES
    X = df_breast.drop(labels=[config.BREAST_CANCER_WISCONSIN_LABEL], axis=1)   
    
#     CONVERT X DATA FRAME TO NUMPY ARRAY
    X = np.array(X)  
    
#     X_feature_names = X.columns
#     print(X_feature_names)
    
#     TRAIN-TEST DATA SPLIT 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    
#     unique, counts = np.unique(y_test, return_counts=True)
#     print(unique, counts)

#     SCALE TO STANDARD SCALER
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
#     CREATE BREAST_CANCER_WISCONSIN_SCALER.PKL FILE
    pkl_directory_file_input = os.path.join(os.path.dirname(__file__), "pkl")            
    pkl_path_file = config.BREAST_CANCER_WISCONSIN_SCALER_PKL
    pkl_directory_path_file = os.path.join(pkl_directory_file_input, pkl_path_file)        
    standard_scaler_pkl = open("".join(pkl_directory_path_file), "wb")
    pickle.dump(scaler, standard_scaler_pkl)
    standard_scaler_pkl.close()    
    
#     SELECT RANDOM FOREST CLASSIFIER MODEL
#     rf_classifier = RandomForestClassifier(n_estimators=1000, max_features="sqrt", criterion="entropy", max_depth=100, bootstrap=False, random_state=1)
#     rf_classifier = RandomForestClassifier()
    rf_classifier = RandomForestClassifier(bootstrap=False, 
                                            criterion="entropy", 
                                            max_depth=85, 
                                            max_features="log2", 
                                            min_samples_leaf=2, 
                                            min_samples_split=5, 
                                            n_estimators=379,
                                            random_state=1)    

# {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 85, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 379}

    print(rf_classifier)
    print()
#     RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
#             oob_score=False, random_state=1, verbose=0, warm_start=False)
    
#     FIT MODEL
    rf_classifier.fit(X_train, y_train)
        
#     CREATE BREAST_CANCER_WISCONSIN_RF_MODEL.PKL FILE
    pkl_directory_file_input = os.path.join(os.path.dirname(__file__), "pkl")            
    pkl_path_file = config.BREAST_CANCER_WISCONSIN_RF_MODEL_PKL
    pkl_directory_path_file = os.path.join(pkl_directory_file_input, pkl_path_file)            
    rf_classifier_pkl = open("".join(pkl_directory_path_file), "wb")
    pickle.dump(rf_classifier, rf_classifier_pkl)
    rf_classifier_pkl.close()    
    
#     PREDICT MODEL
    y_predicted = rf_classifier.predict(X_test)
    
#     CALCULATE ACCURACY SCORE VALUE
    accuracy_score_value = accuracy_score(y_test, y_predicted) * 100
    accuracy_score_value = float("{0:0.2f}".format(accuracy_score_value))    
    print("classification accuracy score:")
    print(accuracy_score_value)
    print()
    
#     TEST CLASS MATRIX
    test_class_matrix = numpy_unique_value_matrix(y_test)
    print("test class matrix:")
    print(test_class_matrix)
    print()
    
#     CALCULATE CONFUSION MATRIX
    confusion_matrix_result = confusion_matrix(y_test, y_predicted)
    print("classification confusion matrix:")
    print(confusion_matrix_result)
    print()
    
#     PLOT CONFUSION MATRIX
    plot_confusion_matrix(confusion_matrix_result, y_unique_class)
    
#     CALCULATE CLASSIFICATION REPORT
    classification_report_result = classification_report(y_test,y_predicted)
    print("classification report:")    
    print(classification_report_result)
    print()  
    
if __name__ == '__main__':
    main()
    
#     C:\Users\Ernest\git\ernest_code\ernest_code\src\breast_cancer_diagnosis_with_keras\csv