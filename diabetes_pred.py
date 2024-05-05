

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from numpy import mean
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
!pip install tensorflow --upgrade
import tensorflow as tf
!pip install keras-tuner --upgrade
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, ActivityRegularization, Dropout
from tensorflow.keras.optimizers import Adam, Adamax, Adagrad, Adafactor, Nadam, Ftrl, Adadelta, AdamW, RMSprop, SGD
from tensorflow.keras.metrics import categorical_crossentropy
import keras_tuner
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report as c_rep
!pip install scikeras
from scikeras.wrappers import KerasClassifier
import torch

df = pd.read_csv('diabetes.csv')
df2 = df.copy()
df2 = df2[['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']]

df.loc[(df['Outcome'] == 0 ) & (df['Insulin'] == 0), 'Insulin'] = 102.5
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'] == 0), 'Insulin'] = 169.5

df.loc[(df['Outcome'] == 0 ) & (df['Glucose'] == 0), 'Glucose'] = 107
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'] == 0), 'Glucose'] = 140

df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'] == 0), 'SkinThickness'] = 27
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'] == 0), 'SkinThickness'] = 32

df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'] == 0), 'BloodPressure'] = 70
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'] == 0), 'BloodPressure'] = 74.5

df.loc[(df['Outcome'] == 0 ) & (df['BMI']==0), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 1 ) & (df['BMI']==0), 'BMI'] = 34.3

from sklearn.model_selection import train_test_split
X = df.drop("Outcome", axis = 1)
std_thing = StandardScaler()
X = std_thing.fit_transform(X)
y = df["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)

def set_bmi(row):
    if row["BMI"] < 18.5:
        return "Under"
    elif row["BMI"] >= 18.5 and row["BMI"] <= 24.9:
        return "Healthy"
    elif row["BMI"] >= 25 and row["BMI"] <= 29.9:
        return "Over"
    elif row["BMI"] >= 30:
        return "Obese"

df = df.assign(BM_DESC=df.apply(set_bmi, axis=1))
df = pd.get_dummies(df, drop_first = True)
def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"
def set_tension(row):
  if row["BloodPressure"] <= 60:
    return "Hypo"
  elif row["BloodPressure"] > 60 and row["BloodPressure"] <= 80:
    return "Normal"
  elif row["BloodPressure"] > 80 and row["BloodPressure"] <= 89:
    return "Hyper1"
  elif row["BloodPressure"] > 89 and row["BloodPressure"] <= 120:
    return "Hyper2"
  else:
    return "Hyper3"
df = df.assign(INSULIN_DESC=df.apply(set_insulin, axis=1))
df = pd.get_dummies(df, drop_first = True, columns = ['INSULIN_DESC'])
df = df.assign(TENSION_DESC=df.apply(set_tension, axis=1))
df = pd.get_dummies(df, drop_first = True, columns = ['TENSION_DESC'])

def set_glucose(row):
  if row["Glucose"] < 70:
    return "Hypoglycemia"
  elif row["BloodPressure"] >= 70 and row["BloodPressure"] <= 100:
    return "Normalg"
  elif row["BloodPressure"] > 100 and row["BloodPressure"] <= 125:
    return "Prediabetes"
  else:
    return "Hyperglycemia"
df = df.assign(GLUCOSE_DESC=df.apply(set_glucose, axis=1))
df = pd.get_dummies(df, drop_first = True, columns = ['GLUCOSE_DESC'])

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
X = df.drop("Outcome", axis = 1)
y = df['Outcome']
stdc = MinMaxScaler()

X = stdc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y)

# models
import lightgbm as lgbm
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
lgbmm = lgbm.LGBMClassifier(colsample_bytree = 0.5062062905660482, max_depth = 5, min_child_samples = 100, min_child_weight = 1, n_estimators = 1500, num_leaves = 17, reg_alpha = 0.1, reg_lambda =0, subsample = 0.6027338166838856)
knn_model = KNeighborsClassifier(metric='manhattan', n_neighbors=5)
rfc_model = RandomForestClassifier(bootstrap=False, max_depth=10, max_features='auto', n_estimators=1000)
logModel = LogisticRegression(C=0.0018329807108324356, penalty='none', solver='saga')
svmm = SVC(C=100, gamma=0.1, probability = True)
gbc_mod = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=200,min_samples_leaf=50,max_depth=5,max_features=17,subsample=0.85,random_state=10, n_estimators = 70)
xgbc = XGBClassifier(learning_rate= 0.09661356050976366, max_depth= 5, n_estimators= 165, subsample= 0.8652062663871867)

from sklearn.model_selection import GridSearchCV
grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
g_res = gs.fit(X_train, y_train)
g_res.best_estimator_

from sklearn.ensemble import StackingClassifier
eclf2 = StackingClassifier(estimators=[('lgb', lgbmm), ('kn', knn_model), ('RF', rfc_model), ('svcc', svmm), ('grad_tree', gbc_mod), ('xgbmm', xgbc)], final_estimator = logModel, stack_method = 'predict_proba')

eclf2.fit(X_train,y_train)

preds = eclf2.predict(X_test)
print(c_rep(y_test, preds))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, preds))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(eclf2, X, y, cv = 10)
print(sum(scores)/len(scores))
