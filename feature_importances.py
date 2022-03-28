#Code used for plotting feature importances for models created in ml_models.py

#feature set as more easily readable names to display in feature importance graphs 

feature_set_readable = ['Age', 'Low MCV', 'Cholesterol',
                 'Low haemoglobin', 'High platelets', 'Constipation (d12)',
                  'Chest Pain (a11)', 'Abdominal Pain (d01)', 'Weight loss (t08)', 'Dysphagia (d21)',
                  'Dysphagia (d21) x2','Reflux (d48)', 'Epigastric pain (d02)', 'Dyspepsia (d07)',
                  'Dyspepsia (d07) x2', 'Nausea/vomiting (d10)', 'Nausea/vomiting (d10) x2',
                  'High LFT', 'High IM', 'High white cell count'
                 ]

#returns a feature importance dataframe with values in ascending order

def fi_as_df(importance_values):
    fi = [] #feature importance

    for i in range(0, len(feature_set_readable)):
        fi.append([feature_set_readable[i], importance_values[i]])

    #feature importance - ascending
    fi_df = pd.DataFrame(fi, columns = ['Feature', 'Score'])
    fi_df = fi_df.sort_values(by=['Score'])
    
    return fi_df

#SVM - feature importances 

#Feature importance - SVM (with linear kernel)
fi_df_svm_lin = fi_as_df(svc_mod_1.coef_[0])

plt.figure(figsize=(20,15))
plt.barh([feature for feature in fi_df_svm_lin['Feature']], [score for score in fi_df_svm_lin['Score']])
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.xlabel('Feature contribution', fontsize = 15, weight = 'bold')
plt.ylabel('Risk factor', fontsize = 15, weight = 'bold')
plt.title('Feature Importance', fontsize = 20)
plt.show()

#Feature importance - SVM (RBF)

#Plot feature importance

importances = permutation_importance(svc_mod_2, data_test, target_test, scoring = 'accuracy')
fi_df_svm_rbf = fi_as_df(importances.importances_mean)

plt.figure(figsize=(20,15))
plt.barh([feature for feature in fi_df_svm_rbf['Feature']], [score for score in fi_df_svm_rbf['Score']])
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.xlabel('Feature contribution', fontsize = 15, weight = 'bold')
plt.ylabel('Risk factor', fontsize = 15, weight = 'bold')
plt.title('Feature Importance', fontsize = 20)
plt.show()

#Feature importance - Logistic Regression

fi_df_logreg = fi_as_df(lr_mod.coef_[0])

plt.figure(figsize=(20,15))
plt.barh([feature for feature in fi_df_logreg['Feature']], [score for score in fi_df_logreg['Score']])
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.xlabel('Feature contribution', fontsize = 15, weight = 'bold')
plt.ylabel('Risk factor', fontsize = 15, weight = 'bold')
plt.title('Feature importance - LogReg', fontsize = 18)
plt.show()

#feature importance - Naïve Bayes

importances = permutation_importance(bnb, data_test_nb, target_test_nb)
fi_df_bnb = fi_as_df(importances.importances_mean)

plt.figure(figsize=(15,10))
plt.barh([feature for feature in fi_df_bnb['Feature']], [score for score in fi_df_bnb['Score']])
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.xlabel('Feature contribution', fontsize = 15, weight = 'bold')
plt.ylabel('Risk factor', fontsize = 15, weight = 'bold')
plt.title('Feature importance', fontsize = 18)
plt.show()

#feature importance - Extreme Gradient Boosting (XGBoost)
fi_df_xgb = fi_as_df(xgb_mod.feature_importances_)

plt.figure(figsize=(15,10))
plt.barh([feature for feature in fi_df_xgb['Feature']], [score for score in fi_df_xgb['Score']])
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.xlabel('Feature contribution', fontsize = 15, weight = 'bold')
plt.ylabel('Risk factor', fontsize = 15, weight = 'bold')
plt.title('Feature importance', fontsize = 18)
plt.show()

#Feature importance - Random Forest
fi_df_rf = fi_as_df(tuned_rf_model.feature_importances_)

#plot feature importance 
plt.figure(figsize=(15,12))
plt.barh([feature for feature in fi_df_rf['Feature']], [score for score in fi_df_rf['Score']])
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.xlabel('Feature contribution', fontsize = 15, weight = 'bold')
plt.ylabel('Risk factor', fontsize = 15, weight = 'bold')
plt.title('Feature importance - Random Forest', fontsize = 20)
plt.show()
