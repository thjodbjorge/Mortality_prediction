#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix, log_loss
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, RFECV, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectFdr
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt
import pickle
# import lifelines as ll
# from lifelines.utils.sklearn_adapter import sklearn_adapter
# CoxRegression = sklearn_adapter(ll.CoxPHFitter, event_col = 'event')
import sys
sys.path.append('/odinn/users/thjodbjorge/Python_functions/')
import Predict_functions as pf
from Calculate_score import calculate_metrics

raw_data = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/raw_with_info.csv',index_col = 'Barcode2d' )
probe_info = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probe_info.csv', index_col = 'SeqId')

pn_info = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/pn_info_Mor/pn_info_Mor_event.csv',index_col = 'Barcode2d' )
probes_to_skip = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probes_to_skip.txt')['probe']


folder = '/odinn/users/thjodbjorge/Proteomics/Mortality2/'
feat_folder = 'Features2/'
pred_folder = 'Predictions3/'


endpoints = ['death']
# endpoints = ['death','Cdeath','Gdeath','Ideath','Jdeath','Otherdeath']
# event_date = event_date_death
time_to_event = pn_info.time_to_death
no_event_before = pn_info.no_death_before
for endpoint in endpoints:
    if endpoint == 'death':
        use_event = pn_info.event_death
        print(use_event.sum())
    elif endpoint == 'Cdeath':
        use_event = pn_info.event_death & (pn_info.ICD_group == 'C')
        print(use_event.sum())
    elif endpoint == 'Gdeath':
        use_event = pn_info.event_death & (pn_info.ICD_group == 'G')
        print(use_event.sum())
    elif endpoint == 'Ideath':
        use_event = pn_info.event_death & (pn_info.ICD_group == 'I')
        print(use_event.sum())
    elif endpoint == 'Jdeath':
        use_event = pn_info.event_death & (pn_info.ICD_group == 'J')
        print(use_event.sum())
    elif endpoint == 'Otherdeath':
        use_event = pn_info.event_death & (~(pn_info.ICD_group == 'C')&~(pn_info.ICD_group == 'G')&~(pn_info.ICD_group == 'I')&~(pn_info.ICD_group == 'J'))
        print(use_event.sum())

y = []
for i in range(1,19):
    y.append(use_event & (time_to_event <= i))

kf = KFold(n_splits=10, random_state=10, shuffle=False) 
I_train_main, I_test_main = train_test_split(pn_info.index, train_size=0.7, random_state = 10)
I_val_main, I_test_main = train_test_split(I_test_main, train_size=0.5, random_state = 10)



file = open(folder+"{}_keep_samples.pkl".format('Mor'),'rb')
keep_samples_dict = pickle.load(file)

print(keep_samples_dict.keys())
# keep_samples_keys = ['Old_18105', 'Old_60105', 'Old_6080','Old_18105_C', 'Old_18105_I', 'Old_18105_J', 'Old_18105_G','Old_18105_Other']
# keep_samples_keys = ['Old_60105','Old_18105','Old_6080']
# keep_samples_keys = ['Old_18105_Neoplasms','Old_18105_I','Old_18105_J','Old_18105_G','Old_18105_Other']
keep_samples_keys = ['Old_18105']


# K = [4, 9]
# K =[14]
K = [1,14]#,0,2,3,5,6,7,8,10,11,12,13]
# K = [13,14]
# K = [6,7,8,10,11,12,13,14]
# K = [4]
for dataset in keep_samples_keys:
    print(dataset)
    keep_samples = keep_samples_dict[dataset]

    I_train = I_train_main.intersection(keep_samples)#.intersection(have_prs)
    I_test = I_val_main.intersection(keep_samples)#.intersection(have_prs)

    print('Training set: {}, MI within 15: {}, 10: {}, 5: {}, 2: {}'.format(len(I_train),y[14][I_train].sum(),y[9][I_train].sum(),y[4][I_train].sum(),y[1][I_train].sum()))
    print('Test set: {}, MI within 15: {}, 10: {}, 5: {}, 2: {}'.format(len(I_test),y[14][I_test].sum(),y[9][I_test].sum(),y[4][I_test].sum(),y[1][I_test].sum()))

        # ### Select data and normalize

    X = np.log(raw_data.iloc[:,16:].drop(probes_to_skip,axis=1))

    all_protein = X.columns
    X['sex'] = pn_info[['sex']].values-1
    X['age'] = pn_info[['Age_at_sample_collection_2']].values

    X['age2'] = X['age']**2
#     X['age3'] = X['age']**3
    X['agesex'] = X['age']*X['sex']
    X['age2sex'] = X['age2']*X['sex']
        
    agesex = ['age','sex','agesex','age2','age2sex']
       
    X['lnage'] = np.log(X['age'])
    X['lnage2'] = X['lnage']**2
    
    X['PAD'] = pn_info['PAD']
    no_bmi = (X['PAD'].isna())
    no_bmi_ind = X[no_bmi].index
    X.loc[I_train.intersection(no_bmi_ind),'PAD'] = X.loc[I_train].PAD.mean()
    X.loc[I_test.intersection(no_bmi_ind),'PAD'] = X.loc[I_test].PAD.mean()
    
    
    X['CAD'] = ~pn_info.no_CAD_before
    X['MI'] = ~pn_info.no_MI_before
    X['cancer'] = pn_info.cancer_margin
    X['ApoB'] = X['SeqId.2797-56']
    X['Smoker'] = pn_info['Smoker'].astype(int).values
    X['diabetes'] = pn_info['T2D'].astype(int).values
    X['HTN_treated'] = pn_info[['HTN_treated']].astype(int).values
#     X['statin'] = pn_info['statin'].astype(int).values
    X['statin'] = pn_info['statin_estimate_unsure'].astype(int).values
    X['ApoBstatin']  = X['ApoB']*X['statin']
    
    X['cancer1y']  = pn_info['cancer1y']
    X['cancer5y']  = pn_info['cancer5y']

    X['GDF15'] = X['SeqId.4374-45'].copy()
    X['GDF152'] = X['GDF15']**2
    X['GDF15age']  = X['GDF15']*X['age']
    X['GDF15sex']  = X['GDF15']*X['sex']
    
    X['bmi'] = pn_info['bmi']

    no_bmi = (X['bmi'].isna())
    no_bmi_ind = X[no_bmi].index
#     X.loc[no_bmi_ind,'bmi'] = X.loc[I_train].bmi.mean()
    X.loc[I_train.intersection(no_bmi_ind),'bmi'] = X.loc[I_train].bmi.mean()       
    X.loc[I_test.intersection(no_bmi_ind),'bmi'] = X.loc[I_test].bmi.mean()   
    
    X['bmi2'] = X['bmi']*X['bmi']
    
    X['Platelets'] = pn_info['Platelets']
    no_p = (X['Platelets'].isna()); print(no_p.sum())
    no_p_ind = X[no_p].index
    X.loc[I_train.intersection(no_p_ind),'Platelets'] = X.loc[I_train].Platelets.mean()
    X.loc[I_test.intersection(no_p_ind),'Platelets'] = X.loc[I_test].Platelets.mean()
    X['Platelets2'] = X['Platelets']*X['Platelets']
    
    X['Creatinine'] = pn_info['Creatinine']
    no_p = (X['Creatinine'].isna()); print(no_p.sum())
    no_p_ind = X[no_p].index
    X.loc[I_train.intersection(no_p_ind),'Creatinine'] = X.loc[I_train].Creatinine.mean()
    X.loc[I_test.intersection(no_p_ind),'Creatinine'] = X.loc[I_test].Creatinine.mean()
    
    X['Triglycerides'] = pn_info['Triglycerides']
    no_p = (X['Triglycerides'].isna()); print(no_p.sum())
    no_p_ind = X[no_p].index
    X.loc[I_train.intersection(no_p_ind),'Triglycerides'] = X.loc[I_train].Triglycerides.mean()    
    X.loc[I_test.intersection(no_p_ind),'Triglycerides'] = X.loc[I_test].Triglycerides.mean()   
    

    X['bmiage'] = X['bmi']*X['age']
    X['bmisex'] = X['bmi']*X['sex']
    X['bmi2age'] = X['bmi2']*X['age']
    X['bmi2sex'] = X['bmi2']*X['sex']
    X['PADage'] = X['PAD']*X['age']
    X['PADsex'] = X['PAD']*X['sex']
    
    X['ApoBage']  = X['ApoB']*X['age']
    X['Smokerage'] = X['Smoker']*X['age']
    X['diabetesage'] = X['diabetes']*X['age'] 
    X['statinage'] = X['statin']*X['age']
    X['CADage'] = X['CAD']*X['age']
    X['MIage'] = X['MI'] * X['age']
    X['HTN_treatedage'] =  X['age']*X['HTN_treated']    
    X['cancerage'] = X['age']*X['cancer']
    
    X['Plateletsage'] = X['Platelets']*X['age']
    X['Creatinineage'] = X['Creatinine']*X['age']
    X['Triglyceridesage'] = X['Triglycerides']*X['age']    
    X['Platelets2age'] = X['Platelets2']*X['age']
    
    X['cancer1yage'] = X['cancer1y']*X['age'] 
    X['cancer5yage'] = X['cancer5y']*X['age']     
    
    X['ApoBsex']  = X['ApoB']*X['sex']
    X['Smokersex'] = X['Smoker']*X['sex']
    X['diabetessex'] = X['diabetes']*X['sex'] 
    X['statinsex'] = X['statin']*X['sex']
    X['CADsex'] = X['CAD']*X['sex']
    X['MIsex'] = X['MI'] * X['sex']
    X['HTN_treatedsex'] =  X['sex']*X['HTN_treated']   
    X['cancersex'] = X['sex']*X['cancer']
    
    X['Plateletssex'] = X['Platelets']*X['sex']
    X['Creatininesex'] = X['Creatinine']*X['sex']
    X['Triglyceridessex'] = X['Triglycerides']*X['sex']        
    
    X = X.join(pd.get_dummies(pn_info['agebin'],drop_first = True,prefix='age'))
    X['ageage2'] = X['age']*X['age_2.0']
    X['ageage3'] = X['age']*X['age_3.0']
    X['ageage4'] = X['age']*X['age_4.0']
    
    agebins = ['age_2.0','age_3.0','age_4.0', 'ageage2','ageage3','ageage4']
    agebinssex = [s+'sex' for s in agebins]
    X[agebinssex] = (X[agebins].transpose()*X['sex']).transpose()    
    
    
    PRS = ['nonHDL_prs', 'HT_prs', 'CAD_prs', 'Cancer_prs', 'Stroke2_prs', 'alz_Jansen',
       'pgc_adhd_2017', 'PD_Nalls_2018', 'edu_160125', 'dep_2018', 'bpd_2018',
       'giant_bmi', 'schizo_clozuk', 'iq_2018', 'ipsych_pgc_aut_2017',
       'pgc_Anorexia_2019']
    X[PRS] = pn_info[PRS]
    
    trad = ['ApoB','Smoker','diabetes','HTN_treated','statin','CAD','MI','bmi','bmi2']
    tradage = ['ApoBage','Smokerage','diabetesage','CADage','MIage','HTN_treatedage','bmiage']
    tradsex = ['ApoBsex','Smokersex','diabetessex','CADsex','MIsex','HTN_treatedsex','bmisex']
    
    tradcoxR = ['Smoker','Smokersex','diabetes','diabetesage','HTN_treated','HTN_treatedage','MI','MIage','CAD','bmi','bmiage','statin','statinage']
    tradextralog = ['Smokersex','diabetessex','CADsex','CADage','MIage','HTN_treatedage','bmiage','statinage','bmi2age','ApoBstatin']
    tradextralognosex = ['CADage','MIage','HTN_treatedage','bmiage']
    tradblood = ['Creatinine','Triglycerides','Platelets','Platelets2','Plateletsage','Creatinineage','Platelets2age']
    tradcancer = ['cancer','cancerage']    
    extra_cancer = ['cancer1y','cancer5y','cancer1yage','cancer5yage']   
    
    X_train = X.loc[I_train]
    X_test = X.loc[I_test]

    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train = (X_train-train_mean)/train_std
    X_test = (X_test-train_mean)/train_std

        ## For survival analysis    
    X_train['event'] = use_event[I_train]
    X_test['event'] = use_event[I_test]

    tte_train = time_to_event[I_train]
    tte_test = time_to_event[I_test]

    ysurv_train = pd.DataFrame()
    ysurv_train['event'] = use_event[I_train]
    ysurv_train['time_to_event'] = time_to_event[I_train]


    for k in K:
        y_train = y[k][I_train]
        y_test = y[k][I_test]
        
        ycsurv_train = pd.DataFrame()
        ycsurv_train['event'] = y_train
        ycsurv_train['time_to_event'] = np.where(tte_train<= (k+1),tte_train,k+1)
        
        X_train['y'] = y_train
        X_test['y'] = y_test

        try:
            
            file = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),'rb')
            features_dict = pickle.load(file)           

#             bor10uni = sorted(features_dict['{}_bor10uni_y{}'.format(dataset,k)])
#             bor10int = sorted(features_dict['{}_bor10int_y{}'.format(dataset,k)])
                        
#             univariate =  sorted(features_dict['{}_uni_cox005'.format(dataset)])
#             bon05 =  sorted(features_dict['{}_uni_cox_bon05'.format(dataset)])
#             bon005 =  sorted(features_dict['{}_uni_cox_bon005'.format(dataset)])
#             fdr005 =  sorted(features_dict['{}_uni_cox_fdr005'.format(dataset)])
#             fdr001 =  sorted(features_dict['{}_uni_cox_fdr001'.format(dataset)])
            
            boruta = sorted(features_dict['{}_boruta_y{}'.format(dataset,k)])
#             cumbor = sorted(features_dict['{}_cumboruta_y{}'.format(dataset,k)])
#             uniboruta = sorted(features_dict['{}_uniboruta_y{}'.format(dataset,k)])
#             cumunibor = sorted(features_dict['{}_cumuniboruta_y{}'.format(dataset,k)])
#             cumunibor15 = sorted(features_dict['{}_cumuniboruta_y14'.format(dataset)])
#             cumunibor10 = sorted(features_dict['{}_cumuniboruta_y9'.format(dataset)])
#             cumbor15 = sorted(features_dict['{}_cumboruta_y14'.format(dataset)])
#             cumbor10 = sorted(features_dict['{}_cumboruta_y9'.format(dataset)])
#             boruta10 = sorted(features_dict['{}_boruta_y9'.format(dataset)])
#             uniboruta10 = sorted(features_dict['{}_uniboruta_y9'.format(dataset)])
            boruta15 = sorted(features_dict['{}_boruta_y14'.format(dataset)])
#             uniboruta15 = sorted(features_dict['{}_uniboruta_y14'.format(dataset)])
            
            
            print('features_loaded')

        except Exception as e:
            print(e)
            print('No features')

        try: 
            file = open(folder+pred_folder+"{}_{}_bootstrap.pkl".format(endpoint,dataset),'rb')
            pred_dict = pickle.load(file)
        except:
            pred_dict = {}
           
        try:
            coef = []
            feat = []
            feat.extend(agesex)
            feat.extend(boruta)
            np.random.seed(10)
            for i in range(1000):
#                 ind = resample(I_train,replace=True)
                
                X_use = X_train.loc[:,feat].sample(frac=1, replace=True)
                y_use = y_train[X_use.index]
                
#                 out = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')

                Cs = np.logspace(-2,0,10)
                model = LogisticRegressionCV(penalty='l1',Cs=Cs,cv = 5,n_jobs=-1, solver ='saga', scoring='neg_log_loss')
                model.fit(X_use,y_use)   
                
                coef.append(model.coef_)
                print(i)
            pred_dict['{}_y{}_asprotein_l1_boot1000'.format(dataset,k)] = coef

            f = open(folder+pred_folder+"{}_{}_bootstrap.pkl".format(endpoint,dataset),"wb")
            pickle.dump(pred_dict,f)
            f.close()
            print('Done k = ',k)
        except Exception as e:
            print('Fail bootstrap')
            print(e)