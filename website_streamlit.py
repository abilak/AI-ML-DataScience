
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


!pip install protobuf==4.21.1
!pip -q install streamlit
!pip -q install pyngrok
import streamlit as st
from pyngrok import ngrok
def launch_website():
  print ("Click this link: ")
  if (ngrok.get_tunnels() != None):
    ngrok.kill()
  public_url = ngrok.connect()
  print (public_url)
  !streamlit run --server.port 80 app.py >/dev/null

!ngrok authtoken 2RJYtp0kbUsf1PjZOCJTfUb72V5_3xUo2ETrBaHzW9A6X74e1
# use colab for this
# Commented out IPython magic to ensure Python compatibility.
# %%writefile header.py
# import streamlit as st
# def create_header():
#   st.title("Diabetes Diagnosis")
#   st.header("Diabetes Diagnosis")
#   st.subheader("By Aaditya B.")
#   st.subheader("WARNING: THIS SHOULD ONLY BE USED AS MEDICAL ADVICE!")

# Commented out IPython magic to ensure Python compatibility.
# %%writefile userinput.py
# import streamlit as st
# import numpy as np
# import joblib
# def get_user_input():
#   pregnancies = st.number_input('How many times have you been pregnant?')
#   glucose = st.number_input('What is your plasma glucose concentration?')
#   bp = st.number_input('What is your diastolic blood pressure (mm Hg)?')
#   skin_t = st.number_input('What is your triceps skin fold thickness (mm)?')
#   insulin = st.number_input('What is your 2-hour serum insulin (mu U/ml)?')
#   bmi = st.number_input('What is your BMI?')
#   dia_ped = st.number_input('What is your Diabetes Pedigree Function?')
#   age = st.number_input('How old are you?')
#   bmi_obese = 0
#   bmi_over = 0
#   bmi_under = 0
#   ins_norm = 0
#   tens_hyper2 = 0
#   tens_hyper3 = 0
#   tens_hypo = 0
#   tens_normal = 0
#   if bmi < 18.5:
#       bmi_under = 1
#   elif bmi >= 18.5 and bmi <= 24.9:
#       pass
#   elif bmi >= 25 and bmi <= 29.9:
#       bmi_over = 1
#   elif bmi >= 30:
#       bmi_obese = 1
#   if insulin >= 16 and insulin <= 166:
#         ins_norm = 1
#   else:
#       ins_norm = 0
# 
#   if bp <= 60:
#     tens_hypo = 1
#   elif bp > 60 and bp <= 80:
#     tens_normal = 1
#   elif bp > 80 and bp <= 89:
#     pass
#   elif bp > 89 and bp <= 120:
#     tens_hyper2 = 1
#   else:
#     tens_hyper3 = 1
# 
#   scale = [0.05882353, 0.00502513, 0.01020408, 0.01010101, 0.00120192, 0.0204499 , 0.42698548, 0.01666667, 1, 1, 1, 1, 1, 1, 1,1]
#   input_features = [[pregnancies*scale[0], glucose*scale[1], bp*scale[2], skin_t*scale[3], insulin*scale[4], bmi*scale[5], dia_ped*scale[6], age*scale[7], bmi_obese, bmi_over, bmi_under, ins_norm, tens_hyper2, tens_hyper3, tens_hypo,tens_normal]] # put your features in here!
#   return input_features

# Commented out IPython magic to ensure Python compatibility.
# %%writefile predictor.py
# import streamlit as st
# import numpy as np
# def make_prediction(model, input_features):
#   return model.predict(input_features)

# Commented out IPython magic to ensure Python compatibility.
# %%writefile response.py
# import streamlit as st
# def get_app_response(prediction):
#   if prediction == 1: # CHANGE THIS!
#     st.write("You have Diabetes")
#   elif prediction == 0:
#     st.write("You do not have diabetes")

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# from header import *
# from userinput import *
# from response import *
# from predictor import *
# import joblib
# import tensorflow as tf
# import keras
# ensembler = joblib.load('ensemble')
# 
# create_header()
# input_features = get_user_input()
# prediction = make_prediction(ensembler, input_features)
# get_app_response(prediction)

launch_website()
