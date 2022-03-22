#code used to implement oesophago-gastric Cancer Risk Assessment Tool (Cancer RAT) [1]

#dictionary representing combinations of symptoms corresponding to risk scores 

RATSdict = {
  "('abn_low_haem',)": 0.002,
  "('abn_hi_plat',)": 0.005,
  "('sym_d12_constipation1',)": 0.002,
  "('sym_a11_chest_pain1',)": 0.002,
  "('sym_d01_abdo_pain1',)": 0.003,
  "('sym_t08_weightloss1',)": 0.009,
  "('sym_d21_dysphagia1',)": 0.048,
  "('sym_d21_dysphagia2',)": 0.055,
  "('sym_d84_reflux1',)": 0.006,
  "('sym_d02_epigastricpain1',)": 0.009,
  "('sym_d07_dyspepsia1',)": 0.007,
  "('sym_d07_dyspepsia2',)": 0.012,
  "('sym_d10_nausea_vomiting1',)": 0.006,
  "('sym_d10_nausea_vomiting2',)": 0.01,
  "('abn_low_haem', 'abn_hi_plat')": 0.006,
  "('abn_low_haem', 'sym_d12_constipation1')": 0.004,
  "('abn_low_haem', 'sym_a11_chest_pain1')": 0.003,
  "('abn_low_haem', 'sym_d01_abdo_pain1')": 0.005,
  "('abn_low_haem', 'sym_t08_weightloss1')": 0.01,
  "('abn_low_haem', 'sym_d21_dysphagia1')": 0.046,
  "('abn_low_haem', 'sym_d84_reflux1')": 0.009,
  "('abn_low_haem', 'sym_d02_epigastricpain1')": 0.016,
  "('abn_low_haem', 'sym_d07_dyspepsia1')": 0.01,
  "('abn_low_haem', 'sym_d10_nausea_vomiting1')": 0.009,
  "('abn_hi_plat', 'sym_d12_constipation1')": 0.009,
  "('abn_hi_plat', 'sym_a11_chest_pain1')": 0.008,
  "('abn_hi_plat', 'sym_d01_abdo_pain1')": 0.008,
  "('abn_hi_plat', 'sym_t08_weightloss1')": 0.018,
  "('abn_hi_plat', 'sym_d21_dysphagia1')": 0.061,
  "('abn_hi_plat', 'sym_d84_reflux1')": 0.016,
  "('abn_hi_plat', 'sym_d02_epigastricpain1')": 0.019,
  "('abn_hi_plat', 'sym_d07_dyspepsia1')": 0.014,
  "('abn_hi_plat', 'sym_d10_nausea_vomiting1')": 0.014,
  "('sym_d12_constipation1', 'sym_a11_chest_pain1')": 0.004,
  "('sym_d12_constipation1', 'sym_d01_abdo_pain1')": 0.004,
  "('sym_d12_constipation1', 'sym_t08_weightloss1')": 0.011,
  "('sym_d12_constipation1', 'sym_d21_dysphagia1')": 0.042,
  "('sym_d12_constipation1', 'sym_d84_reflux1')": 0.007,
  "('sym_d12_constipation1', 'sym_d02_epigastricpain1')": 0.014,
  "('sym_d12_constipation1', 'sym_d07_dyspepsia1')": 0.008,
  "('sym_d12_constipation1', 'sym_d10_nausea_vomiting1')": 0.006,
  "('sym_a11_chest_pain1', 'sym_d01_abdo_pain1')": 0.003,
  "('sym_a11_chest_pain1', 'sym_t08_weightloss1')": 0.011,
  "('sym_a11_chest_pain1', 'sym_d21_dysphagia1')": 0.058,
  "('sym_a11_chest_pain1', 'sym_d84_reflux1')": 0.006,
  "('sym_a11_chest_pain1', 'sym_d02_epigastricpain1')": 0.009,
  "('sym_a11_chest_pain1', 'sym_d07_dyspepsia1')": 0.007,
  "('sym_a11_chest_pain1', 'sym_d10_nausea_vomiting1')": 0.006,
  "('sym_d01_abdo_pain1', 'sym_t08_weightloss1')": 0.014,
  "('sym_d01_abdo_pain1', 'sym_d21_dysphagia1')": 0.065,
  "('sym_d01_abdo_pain1', 'sym_d84_reflux1')": 0.006,
  "('sym_d01_abdo_pain1', 'sym_d02_epigastricpain1')": 0.009,
  "('sym_d01_abdo_pain1', 'sym_d07_dyspepsia1')": 0.01,
  "('sym_d01_abdo_pain1', 'sym_d10_nausea_vomiting1')": 0.007,
  "('sym_t08_weightloss1', 'sym_d21_dysphagia1')": 0.055,
  "('sym_t08_weightloss1', 'sym_d84_reflux1')": 0.031,
  "('sym_t08_weightloss1', 'sym_d02_epigastricpain1')": 0.042,
  "('sym_t08_weightloss1', 'sym_d07_dyspepsia1')": 0.021,
  "('sym_t08_weightloss1', 'sym_d10_nausea_vomiting1')": 0.028,
  "('sym_d21_dysphagia1','sym_d84_reflux1')": 0.05,
  "('sym_d21_dysphagia1', 'sym_d02_epigastricpain1')": 0.093,
  "('sym_d21_dysphagia1', 'sym_d07_dyspepsia1')": 0.098,
  "('sym_d21_dysphagia1', 'sym_d10_nausea_vomiting1')": 0.073,
  "('sym_d84_reflux1', 'sym_d02_epigastricpain1')": 0.015,
  "('sym_d84_reflux1', 'sym_d07_dyspepsia1')": 0.009,
  "('sym_d84_reflux1', 'sym_d10_nausea_vomiting1')": 0.023,
  "('sym_d02_epigastricpain1', 'sym_d07_dyspepsia1')": 0.014,
  "('sym_d02_epigastricpain1', 'sym_d10_nausea_vomiting1')": 0.013,
  "('sym_d07_dyspepsia1', 'sym_d10_nausea_vomiting1')": 0.013,
  "[]": 0.00,
}


RATs_probs = np.zeros(10087,)

#find all subsets of size s from n elements

def findsubsets(s, n):
    return list(itertools.combinations(s, n))


#generate a risk score given an individual observation to test

def generate_risk_score(test_element):
    listofsets = []
    riskscorelist = []
    riskscores = []
#test element is a dataframe according to one observation 'to test'

    if test_element.loc['age_group'].vals == 1.0:
        
#get all symptoms
        to_test_subset = test_element.loc[test_element['vals'] == 1]
    
#get the names of these rows
        indexNamesArr = to_test_subset.index.values
    
#get all subsets of symptoms size 1 and 2 

        subsetssize2 = findsubsets(indexNamesArr, 2)
        subsetssize1 = findsubsets(indexNamesArr, 1)

        for s2 in subsetssize2:
            listofsets.append(s2)

        for s1 in subsetssize1:
            listofsets.append(s1)
        
#collect all risk scores according to pairwise combinations of symptoms

        for x in listofsets:
            s = str(x)
            riskscorelist.append(RATSdict.get(s))
    
#get rid of none values 
    
        riskscores = [i for i in riskscorelist if i]
    riskscores.append(0.00)
    
    return(max(riskscores))

#generate risk scores for each element in the test dataset

for i in range(0, len(data_test)):
    to_test = data_test.iloc[i,[0,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
    to_test = to_test.to_frame()
    to_test.rename(columns = {list(to_test)[0]: 'vals'}, inplace = True)
    risk_score = generate_risk_score(to_test)
    
    RATs_probs[i] = risk_score
    
#generate predictions from risk scores

threshold = 0.02 #set risk prediction threshold - 0.02 is the recommended threshold 

#final predictions 
RATs_preds = (RATs_probs >= threshold).astype('int')

#References
# [1] Stapley S, Peters TJ, Neal RD, Rose PW, Walter FM, Hamilton W. The risk of oesophago-gastric cancer in symptomatic patients in primary care: a large case–control study using electronic records. Br J Cancer. 2013 Jan;108(1):25–31. 