import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("emails.csv")

print(data.head())

print(data.describe())

print(data.shape)

data_drop_col=data.drop('Email No.', axis=1)

x=data.drop(columns=['Email No.', 'Prediction'])
y=data['Prediction']
x_numeric=x.select_dtypes(include=['float64','int64'])
scale=StandardScaler()
x_scale=scale.fit_transform(x_numeric)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=44)


lr = LogisticRegression(max_iter=1000)
svm= SVC(kernel='linear', random_state=44)
dt= DecisionTreeClassifier()
rf=RandomForestClassifier()
gb=GradientBoostingClassifier()


diff_models= [lr, svm, dt, rf, gb]
for model in diff_models:
    model.fit(x_train, y_train)
    Accuracy= model.score(x_test, y_test)
    print(f"{model} Accuracy: {Accuracy}")


def model_evaluation(model, x_test, y_test):
    y_pred = model.predict(x_test)
    Accuracy =accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred)
    Recall= recall_score(y_test, y_pred)
    f1Score= f1_score(y_test)
    print(f"Accuracy: {Accuracy}")
    print(f"Precision: {Precision}")
    print(f"Recall: {Recall}")
    print(f"f1 Score: {f1Score}")
select_model=rf
model_evaluation(select_model, x_test,y_test)

models_evaluation = [lr,svm, dt, rf, gb]
for model in models_evaluation:
    print(f"{model} Evaluation: ")
    model_evaluation(model, x_test,y_test)

#Cross-Validation Score
for model in diff_models:
    cv_score = cross_val_score(model, x_scale, y, cv=5)
    print(f"{model}: Cross-Validation Score: {cv_score}")
    print(f"{model}: Mean of Cross-Validation: {cv_score.mean()}")