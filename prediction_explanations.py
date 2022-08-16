#Individual predictions were generated using the Local Interpretable Model-Agnostic Explanations (LIME) package [1]
#to be run after constructing the models as in ml_models.py

from lime.lime_tabular import LimeTabularExplainer

np.random.seed(1)

#use test, train data from previous, convert to np format to generate prediction explanations

data_train_np = data_train.to_numpy()
data_test_np = data_test.to_numpy()
target_train_np = target_train.to_numpy()
target_test_np = target_test.to_numpy()

#do the same but for naïve bayes training and test data, which is slightly different 
data_train_nb_np = data_train_nb.to_numpy()
data_test_nb_np = data_test_nb.to_numpy()
target_train_nb_np = target_train_nb.to_numpy()
target_test_nb_np = target_test_nb.to_numpy()

class_names = ['No cancer', 'Cancer']


feature_names_r = ['Age', 'Low MCV',
                 'Cholesterol', 'Low Haemoglobin', 'High Platelets', 'Constipation (d12)',
                  'Chest Pain (a11)', 'Abdominal Pain (do1)', 'Weight loss (t08)', 'Dysphagia (d21)',
                   'Dysphagia x2 (d21)','Reflux (d28)', 'Epigastric Pain (d02)', 'Dyspepsia (d07)',
                   'Dyspepsia x2 (d07)', 'Nausea/Vomiting (d10)', 'Nausea/Vomiting x2 (d10)',
                  'High LFT', 'High IM', 'High White Cell Count'
                 ]

feature_names_nb = ['Age', 'Low MCV', 'High cholesterol', 'Low Haemoglobin', 'High Platelets', 'Constipation (d12)',
                  'Chest Pain (a11)', 'Abdominal Pain (do1)', 'Weight loss (t08)', 'Dysphagia (d21)',
                   'Dysphagia x2 (d21)','Reflux (d28)', 'Epigastric Pain (d02)', 'Dyspepsia (d07)',
                   'Dyspepsia x2 (d07)', 'Nausea/Vomiting (d10)', 'Nausea/Vomiting x2 (d10)',
                  'High LFT', 'High IM', 'High White Cell Count'
                 ]

#create explainer 
explainer = LimeTabularExplainer(data_train_np, class_names=class_names, feature_names = feature_names_r)

#SVM (linear) explainer 

idx = 1 #select individual observation to inspect 
thresh = threshold #choose threshold at which to observe prediction
svm_exp = explainer.explain_instance(data_test_np[idx], clf.predict_proba, num_features=num_features)
svm_probab = clf.predict_proba([data_test_np[idx]])[0,1]
svm_predict = (svm_probab >= thresh).astype('int')

print('Document id: %d' % idx)
print('Probability(cancer) =', svm_probab)
print('Classified as (Threshold = ' + str(thresh) + '):', class_names[svm_predict])
print('True class: %s' % class_names[int(target_test_np[idx])])

#for displaying prediction visual in notebook
svm_exp.show_in_notebook(show_table=True, show_all=True)

#SVM (RBF) explainer 

idx = 1 #select individual observation to inspect 
thresh = threshold #choose threshold at which to obersve prediction 
svm_rbf_exp = explainer.explain_instance(data_test_np[idx], clf2.predict_proba, num_features=num_features)
svm_rbf_probab = clf2.predict_proba([data_test_np[idx]])[0,1]
svm_rbf_predict = (svm_rbf_probab >= thresh).astype('int')

print('Document id: %d' % idx)
print('Probability(cancer) =', svm_rbf_probab)
print('Classified as (Threshold = ' + str(thresh) + '):', class_names[svm_rbf_predict])
print('True class: %s' % class_names[int(target_test_np[idx])])

#for displaying prediction visual in notebook
svm_rbf_exp.show_in_notebook(show_table=True, show_all=True)

#Random Forest explainer

idx = 1 #select individual observation to inspect
thresh = threshold #choose threshold at which to observe prediction
rf_exp = explainer.explain_instance(data_test_np[idx], tuned_model.predict_proba, num_features=num_features)
forest_prob = tuned_model.predict_proba([data_test_np[idx]])[0,1]
forest_pred = (forest_prob >= thresh).astype('int')

print('Document id: %d' % idx)
print('Probability(cancer) =', forest_prob)
print('Classified as (Threshold = ' + str(thresh) + '):', class_names[forest_pred])
print('True class: %s' % class_names[int(target_test_np[idx])])

#for displaying prediction visual in notebook
rf_exp.show_in_notebook(show_table=True, show_all=True)

#Logistic Regression explainer 

idx = 1 #select individual observation to inspect 
thresh = threshold #choose threshold at which to observe prediction 
lr_exp = explainer.explain_instance(data_test_np[idx], lr_mod.predict_proba, num_features=num_features)
lr_probab = lr_mod.predict_proba([data_test_np[idx]])[0,1]
lr_predict = (lr_probab >= thresh).astype('int')

print('Document id: %d' % idx)
print('Probability(cancer) =', lr_probab)
print('Classified as (Threshold = ' + str(thresh) + '):', class_names[lr_predict])
print('True class: %s' % class_names[int(target_test_np[idx])])

#for displaying prediction visual in notebook
lr_exp.show_in_notebook(show_table=True, show_all=True)

#XGBoost explainer 

xgb_mod.fit(data_train_np, target_train_np) #fit model to np array version of training data

idx = 1 #select individual observation to inspect
thresh = threshold #choose threshold at which to observe prediction 
xgb_exp = explainer.explain_instance(data_test_np[idx], xgb_mod.predict_proba, num_features=num_features)
xgb_prob = xgb_mod.predict_proba([data_test_np[idx]])[0,1]
xgb_pred = (xgb_prob >= thresh).astype('int')

print('Document id: %d' % idx)
print('Probability(cancer) =', xgb_prob)
print('Classified as (Threshold = ' + str(thresh) + '):', class_names[xgb_pred])
print('True class: %s' % class_names[int(target_test_np[idx])])

#for displaying prediction visual in notebook
xgb_exp.show_in_notebook(show_table=True, show_all=True)

#Naive Bayes explainer 

#create explainer 
explainer_nb = LimeTabularExplainer(data_train_nb_np, class_names=class_names, feature_names = feature_names_nb)

bnb.fit(data_train_nb_np, target_train_nb_np) #fit model to np array version of training data

idx = 1 #select individual observation to inspect
thresh = threshold #choose threshold at which to observe prediction
bnb_exp = explainer.explain_instance(data_test_nb_np[idx], bnb.predict_proba, num_features=num_features)
bnb_prob = bnb.predict_proba([data_test_nb_np[idx]])[0,1]
bnb_pred = (bnb_prob >= thresh).astype('int')

print('Document id: %d' % idx)
print('Probability(cancer) =', bnb_prob)
print('Classified as (Threshold = ' + str(thresh) + '):', class_names[bnb_pred])
print('True class: %s' % class_names[int(target_test_nb_np[idx])])

#for displaying prediction visual in notebook
bnb_exp.show_in_notebook(show_table=True, show_all=True)


#References 
#[1] Ribeiro MT, Singh S, Guestrin C. ‘Why Should I Trust You?’: Explaining the Predictions of Any Classifier. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining [Internet]. San Francisco California USA: ACM; 2016 [cited 2021 Aug 12]. p. 1135–44. Available from: https://dl.acm.org/doi/10.1145/2939672.2939778
