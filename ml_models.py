#required packages 

import pandas as pd
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
import lime
import lime.lime_tabular
import xgboost as xgb
import scipy as scipy

from sklearn import metrics, svm, preprocessing, neural_network
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, balanced_accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from numpy.random import randn, seed
from scipy import stats
from scipy.stats import pearsonr
from matplotlib.gridspec import GridSpec

og_data = #insert dataset here 

np.random.seed(33)


#####################################
## List of feature set experiments ##
#####################################

#Just symptoms from RATs matrix 
feature_set_1 = ['abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1', 'sym_d10_nausea_vomiting1'
                 ]

#Symptoms from RATs matrix inc. age
feature_set_1a = ['age_group', 'abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1', 'sym_d10_nausea_vomiting1'
                 ]

#Symptoms from RATs matrix including repeat presentations and age
feature_set_1b = ['age_group', 'abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d21_dysphagia2','sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1',
                  'sym_d07_dyspepsia2', 'sym_d10_nausea_vomiting1', 'sym_d10_nausea_vomiting2'
                 ]

#Symptoms from RATs matrix with additional variables for cholesterol and age (inc. repeats)
feature_set_1c = ['age_group', 'chol_sq',
                 'abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d21_dysphagia2','sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1',
                  'sym_d07_dyspepsia2', 'sym_d10_nausea_vomiting1', 'sym_d10_nausea_vomiting2'
                 ]

#all 16 risk factors
feature_set_2 = ['abn_low_MCV',
                 'abn_hi_chol', 'abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1', 'sym_d10_nausea_vomiting1',
                  'abn_hi_LFT', 'abn_hi_IM', 'abn_hi_wcc'
                 ]


#all 16 risk factors including age 
feature_set_2a = ['abn_low_MCV', 'age_group',
                 'abn_hi_chol', 'abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1', 'sym_d10_nausea_vomiting1',
                  'abn_hi_LFT', 'abn_hi_IM', 'abn_hi_wcc'
                 ]

#all 16 risk factors including repeat presentations and age 
feature_set_2b = ['abn_low_MCV', 'age_group',
                 'abn_hi_chol', 'abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d21_dysphagia2','sym_d84_reflux1', 'sym_d02_epigastric_pain1',
                  'sym_d07_dyspepsia1', 'sym_d07_dyspepsia2', 'sym_d10_nausea_vomiting1',
                  'sym_d10_nausea_vomiting2', 'abn_hi_LFT', 'abn_hi_IM', 'abn_hi_wcc'
                 ]

#All 16 risk factors including cholesterol (as continuous value), repeats and age
#(this was optimal/final subset)
feature_set_2c = ['age_group', 'abn_low_MCV', 'chol_sq',
                 'abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d21_dysphagia2','sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1',
                  'sym_d07_dyspepsia2', 'sym_d10_nausea_vomiting1', 'sym_d10_nausea_vomiting2',
                  'abn_hi_LFT', 'abn_hi_IM', 'abn_hi_wcc'
                 ]

#all possible symptoms & lab test results (includes repeats and age)
feature_set_3 = ['age_group', 'abn_low_MCV',
                 'chol_sq','abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d21_dysphagia2','sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1',
                  'sym_d07_dyspepsia2', 'sym_d10_nausea_vomiting1', 'sym_d10_nausea_vomiting2',
                  'sym_a04_fatigue1', 'sym_r02_short_of_breath1', 'sym_r05_cough1',
                  'abn_hi_LFT', 'abn_hi_IM', 'abn_hi_wcc'
                 ] 

#optimal feature set 
feature_set = feature_set_2c

target_feature = 'case_control'

############################
## Dataset pre-processing ##
############################

#Convert categorical into binary for age groups

og_data['age_group'].replace({'Under 55':0.0, '55 and over':1.0}, inplace=True)
og_data['age_band'].replace({'Under 60':0.0, '60 and over':1.0}, inplace=True)


#Normalisation of continuous variables

#normalise ages
min_max_scaler = preprocessing.MinMaxScaler()
ages_scaled = min_max_scaler.fit_transform(og_data[['age_at_dx']])
og_data['age_at_dx'] = ages_scaled

#normalise cholesterol
min_max_scaler2 = preprocessing.MinMaxScaler()
chol_sq_scaled = min_max_scaler2.fit_transform(og_data[['chol_sq']])
og_data['chol_sq'] = chol_sq_scaled

#split into data and target
data = og_data[feature_set]
target = og_data[target_feature]
num_features = len(feature_set)

#split into train and test
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.25, shuffle = True)

#check shapes
print('Training data shape: ', data_train.shape)
print('Test data shape: ', data_test.shape)
print('Training labels shape: ', target_train.shape)
print('Testing labels shape: ', target_test.shape)


#The code below shows the creation of the ML-based probabilistic classifiers. This represents the final versions of these models. Please see the supplementary document for further details. 


#######################
## Model: Linear SVM ##
#######################

threshold = 0.6 #set risk prediction threshold

svm = LinearSVC()
clf = CalibratedClassifierCV(svm) 
clf.fit(data_train, target_train)
test_probas = clf.predict_proba(data_test)
test_preds = (test_probas[:,1] >= threshold).astype('int')

#print results

print("Linear SVM with risk threshold "+ str(threshold) + ":")
print('')
print("Accuracy:",metrics.accuracy_score(target_test, test_preds))
print("AUC/ROC score:", metrics.roc_auc_score(target_test, test_probas[:,1]))
print('')
print("Precision:",metrics.precision_score(target_test, test_preds))
print("Recall:",metrics.recall_score(target_test, test_preds))
print('')
print("F1 score:",metrics.f1_score(target_test, test_preds))

#Grid search CV for optimising hyperparameters

svc = SVC()
# defining parameter range
param_grid_svm = {'C': [0.1, 1, 10, 100], 
              'gamma': [0.001, 0.01, 0.1, 1],
              'kernel': ['rbf', 'sigmoid', 'poly', 'linear']} 
  
gridsearched = GridSearchCV(svc, param_grid_svm, refit = True, verbose = 3)
  
# fitting the model for grid search
gridsearched.fit(data_train, target_train)
gridsearched.best_params_
#result: C = 0.1, gamma = 1, kernel = 'rbf'

#############################
## Model: SVM (RBF kernel) ## 
#############################

#SVM following hyperparameter tuning 

threshold = 0.525 #set risk prediction threshold
svm2 = SVC(C = 0.1, gamma = 1, kernel = 'rbf')
clf2 = CalibratedClassifierCV(svm2) 
clf2.fit(data_train, target_train)
test_probas2 = clf2.predict_proba(data_test)
test_preds2 = (test_probas2[:,1] >= threshold).astype('int')

#print results

print('MODEL: SVM (RBF kernel), RISK THRESHOLD: ' + str(threshold))
print('')
print("Accuracy:",metrics.accuracy_score(target_test, test_preds2))
print("AUC/ROC score:", metrics.roc_auc_score(target_test,test_probas2[:,1]))
print('')
print("Precision:",metrics.precision_score(target_test, test_preds2))
print("Recall:",metrics.recall_score(target_test, test_preds2))
print('')
print("F1 score:",metrics.f1_score(target_test, test_preds2))

################################
## Model: Logistic Regression ##
################################

#Hyperparameter tuning using grid search cross-validation

param_grid_lr = {
        'solver': [‘lbfgs’, ‘liblinear’, ‘sag’],
        'C': [0.01, 0.1, 1, 10, 100, 1000]
        }

lr_classifier = LogisticRegression()
grid_search = GridSearchCV(estimator = lr_classifier, param_grid = param_grid_lr, 
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search.fit(data_train, target_train)
grid_search.best_params_
#result: 'solver' = 'sag', 'C' = 1

#initialise logistic regression model 
lr_mod = LogisticRegression(random_state=0, solver= 'sag', C=1)


threshold = 0.8 #set risk threshold
lr_mod.fit(data_train, target_train)

test_probabs = lr_mod.predict_proba(data_test)
test_predicts = (test_probabs[:,1] >= threshold).astype('int')

#print results

print('MODEL: LogReg, RISK THRESHOLD: ' + str(threshold))
print('')
print("Accuracy:",metrics.accuracy_score(target_test, test_predicts))
print("AUC/ROC:", metrics.roc_auc_score(target_test, test_probabs[:,1]))
print('')
print("Precision:",metrics.precision_score(target_test, test_predicts))
print("Recall:",metrics.recall_score(target_test, test_predicts))
print('')
print("F1 score:", metrics.f1_score(target_test, test_predicts))

##########################
## Model: Random Forest ##
##########################

#Hyperparameter tuning using grid search cross-validation

param_grid_rf = {'bootstrap': [True],
 'max_depth': [25, 50, 75, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [2, 4],
 'min_samples_split': [2, 5, 10],
 'criterion': ['entropy'],          
 'n_estimators': [100, 150, 200, 250, 500]}

rf_classifier = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf_classifier, param_grid = param_grid_rf, 
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search.fit(data_train, target_train)
grid_search.best_params_
#result: 'bootstrap'= True, 'max_depth'= 75,'max_features' = 'auto', 'min_samples_leaf' = 4, 'min_samples_split' = 5, 'criterion' = 'entropy','n_estimators' = 250


#create model
tuned_model = RandomForestClassifier(n_estimators = 250,
                                    bootstrap = True,
                                    criterion = 'entropy',
                                    max_features = 'auto',
                                    min_samples_leaf = 4,
                                    min_samples_split = 5,
                                    max_depth = 75 
                                    )

#fit to entire training dataset 
tuned_model.fit(data_train, target_train)

#predictions for each class
rf_predictions = tuned_model.predict(data_test)
# Probabilities for each class
rf_probs = tuned_model.predict_proba(data_test)[:, 1]

threshold = 0.7 #set risk prediction threshold
ranf_probs = tuned_model.predict_proba(data_test)
predicted = (ranf_probs[:,1] >= threshold).astype('int')

#performance on final test set
accuracy = accuracy_score(target_test, predicted)
roc_value = roc_auc_score(target_test, predicted)
roc_value_2 = roc_auc_score(target_test, ranf_probs[:,1])
precision= precision_score(target_test, predicted)
recall = recall_score(target_test, predicted)
f1 = f1_score(target_test, predicted)

#print results 

print('MODEL: Random Forest, RISK THRESHOLD: ' + str(threshold))
print('')
print('Accuracy:', accuracy)
print('AUC/ROC:', roc_value)
print('')
print('Precision:', precision)
print('Recall', recall)
print('')
print('F1 score:', f1)

##################################
## Model: Bernoulli Naïve Bayes ##
##################################

#Feature set for BNB slightly different - need all binary features 

feature_set_nb = ['age_group', 'abn_low_MCV', 'abn_hi_chol',
                 'abn_low_haem', 'abn_hi_plat', 'sym_d12_constipation1',
                  'sym_a11_chest_pain1', 'sym_d01_abdo_pain1', 'sym_t08_weightloss1', 'sym_d21_dysphagia1',
                  'sym_d21_dysphagia2','sym_d84_reflux1', 'sym_d02_epigastric_pain1', 'sym_d07_dyspepsia1',
                  'sym_d07_dyspepsia2', 'sym_d10_nausea_vomiting1', 'sym_d10_nausea_vomiting2',
                  'abn_hi_LFT', 'abn_hi_IM', 'abn_hi_wcc'
                 ]

data_nb = og_data[feature_set_nb]
target_nb = og_data[target_feature]
num_features = len(feature_set_nb)

#split into train and test
data_train_nb, data_test_nb, target_train_nb, target_test_nb = train_test_split(data_nb, target_nb, test_size = 0.25, shuffle = True)


threshold = 0.5 #set risk prediction threshold

#create bnb model
bnb = BernoulliNB()

#train the algorithm on training data and make predictions on test set
bnb.fit(data_train_nb, target_train_nb)

threshold = 0.725 #set risk prediction threshold

bnb_test_probs = bnb.predict_proba(data_test_nb)
bnb_test_predicts = (bnb_test_probs[:,1] >= threshold).astype('int')

#print results

print('MODEL: Bernoulli Naïve Bayes, RISK THRESHOLD: ' + str(threshold))
print('')
print("Accuracy:",metrics.accuracy_score(target_test_nb, bnb_test_predicts))
print("AUC/ROC:", metrics.roc_auc_score(target_test_nb, bnb_test_probs[:,1]))
print('')
print("Precision:",metrics.precision_score(target_test_nb, bnb_test_predicts))
print("Recall:",metrics.recall_score(target_test_nb, bnb_test_predicts))
print('')
print("F1 score:", metrics.f1_score(target_test_nb, bnb_test_predicts))

####################
## Model: XGBoost ##
####################

#Hyperparameter tuning using grid search cross-validation

xgbmodel = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False)

# defining parameter range
param_grid_xgb = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 2, 5],
        'subsample': [0.5, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
  
gridsearched = GridSearchCV(xgbmodel, param_grid_xgb, refit = True, verbose = 3, cv = 5)
  
# fitting the model for grid search
gridsearched.fit(data_train, target_train)

gridsearched.best_params_
#result: 'min_child_weight' = 10, 'gamma' = 5, 'subsample' = 0.8, 'colsample_bytree' = 0.5, 'max_depth' = 4

threshold=0.7 #set risk prediction threshold

#initialise XGBoost model 
xgb_mod = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, colsample_bytree=0.5, gamma=5, max_depth=4, min_child_weight=10, subsample=0.8)
xgb_mod.fit(data_train, target_train)

test_probs_xgb = xgb_mod.predict_proba(data_test)
test_pred_xgb = (test_probs_xgb[:,1] >= threshold).astype('int')


#print results
print('Model: XGBoost, RISK THRESHOLD: ' + str(threshold))
print('')
print("Accuracy:",metrics.accuracy_score(target_test, test_pred_xgb))
print("AUC/ROC:", metrics.roc_auc_score(target_test, test_probs_xgb[:,1]))
print('')
print("Precision:",metrics.precision_score(target_test, test_pred_xgb))
print("Recall:",metrics.recall_score(target_test, test_pred_xgb))
print("F1 score:", metrics.f1_score(target_test, test_pred_xgb))
