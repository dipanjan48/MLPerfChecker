import pandas as pd
import numpy as np
from preprocessing import preprocessing
from model import ModelSelection

df = pd.read_csv("C:/Users/Dipanjan/Desktop/train1.csv")
df_sub = pd.read_csv("C:/Users/Dipanjan/Desktop/test.csv")
var_d = 'Survived'

preproc = preprocessing(df , var_d)

X_train, X_test , Y_train , Y_test = preproc.split()
print (X_train)

model = ModelSelection(X_train, X_test , Y_train , Y_test)

cm, precision_score1 , recall_score1 = model.ModelPerformance()

print (cm, precision_score1 , recall_score1)

