# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 07:00:25 2020

@author: JAI KRISHNA
"""
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('Data.csv')
Ind_v = dataset.iloc[:,:-1].values
Dep_v = dataset.iloc[:,3].values

#Preprocessing Data

#Elimination of Missing Variable using mean

from sklearn.impute import SimpleImputer
mis = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose =0 )

        # mis = mis.fit(X[:,1:3])
        # Ind_v[:,1:3] = mis.transform(X[:,1:3])
        # fit_transform can be used when working with the same dataset

Ind_v[:,1:3] = mis.fit_transform(Ind_v[:,1:3])

#Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('Encoder', OneHotEncoder(), [0])], remainder='passthrough')
Ind_v = np.array(ct.fit_transform(Ind_v), dtype=float)

        # label_Ind = LabelEncoder()
        # Ind_v[:,0] =label_Ind.fit_transform(Ind_v[:,0])
        # onehot = OneHotEncoder(categorical_features=[0])
        # Ind_v = onehot.fit(Ind_v).toarray()

labelencoder = LabelEncoder()
Dep_v = labelencoder.fit_transform(Dep_v)

#Spliting Data set
from sklearn.model_selection import train_test_split
Ind_vtrain,Ind_vtest,Dep_vtrain,Dep_vtest = train_test_split(Ind_v,Dep_v, test_size=0.2, random_state=0)
