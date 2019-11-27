
import numpy as np
import pandas as pd 
import pandas_profiling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def main():  
#     load diabetes.csv file
    df_diabetes = pd.read_csv("diabetes.csv")
    
#     get df information 
    df_diabetes.info()
    print()

#     data profiling            
#     profile = df_diabetes.profile_report(title="Pima Indians Diabetes Project Report")
#     profile.to_file(output_file="Pima Indians Diabetes Project Report.html")
#     exit()
    
#     define feature columns
    X = df_diabetes.drop("Class", axis=1)
    X = np.array(X)
    
#     define label
    y = df_diabetes["Class"]
    y =  np.array(y)

#     data split (20% test and 80% train)
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=1)
    
#     data standard scaling
    scaler = StandardScaler()    
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
#     define ml model to use
#     model_classifier = RandomForestClassifier()
    model_classifier = xgb.XGBClassifier()
    
    print(model_classifier)
    print()
   
#     model fitting
    model_classifier.fit(X_train, y_train)
    
#     model prediction
    y_predict = model_classifier.predict(X_test)    
     
#     calculate classification accuracy score
    accuracy_score_value = accuracy_score(y_test, y_predict) * 100
    accuracy_score_value = float("{0:0.2f}".format(accuracy_score_value))    
    print("Classification Accuracy Score: {} %".format(accuracy_score_value))
    print()
     
if __name__ == '__main__':
    main()