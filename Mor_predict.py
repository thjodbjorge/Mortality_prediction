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
import sys
sys.path.append('/odinn/users/thjodbjorge/Python_functions/')
import Predict_functions as pf
from Calculate_score import calculate_metrics

raw_data = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/raw_with_info.csv',index_col = 'Barcode2d' )
probe_info = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probe_info.csv', index_col = 'SeqId')

pn_info = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/pn_info_Mor/pn_info_Mor_event.csv',index_col = 'Barcode2d' )
probes_to_skip = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probes_to_skip.txt')['probe']
nopro = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/no_protein_probes.txt', header = None)[0] # non-proten probes that were included 
probes_to_skip = set(probes_to_skip).union(set(nopro))

folder = '/odinn/users/thjodbjorge/Proteomics/Mortality2/'
feat_folder = 'Features2/'
pred_folder = 'Predictions4/'


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
keep_samples_keys = ['Old_18105','Old_60105','Old_18105_Neoplasms','Old_18105_I','Old_18105_J','Old_18105_G','Old_18105_Other']
# keep_samples_keys = ['Old_no_comor_18105','Old_no_comor_60105']
# keep_samples_keys = ['Old_18105_Other']
# keep_samples_keys = ['Old_18105','Old_60105']
# keep_samples_keys = ['Old_cardio_risk_18105']


# K = [4, 9]
# K =[14]
K = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# K = [9]
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
    X['PCE'] = ~pn_info.no_PCEend_before
    X['Stroke'] = ~pn_info.no_stroke_before
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
    
    # baseline
    trad = ['ApoB','Smoker','diabetes','HTN_treated','statin','CAD','MI','bmi','bmi2']
    tradextralog = ['Smokersex','diabetessex','CADsex','CADage','MIage','HTN_treatedage','bmiage','statinage','bmi2age','ApoBstatin']
    tradcancer = ['cancer','cancerage']    
    
    tradextralognosex = ['CADage','MIage','HTN_treatedage','bmiage']
    #baseline2
    trad2 = ['ApoB','Smoker','diabetes','HTN_treated','statin','CAD','MI','bmi','bmi2','Stroke','Smokersex',
             'diabetessex','CADsex','CADage','MIage','HTN_treatedage','bmiage','statinage','bmi2age','ApoBstatin',
            'cancer','cancerage','cancersex']
    tradlifestyle = ['ApoB','Smoker','diabetes','HTN_treated','statin','bmi','bmi2','Smokersex',
                     'diabetessex','HTN_treatedage','bmiage','statinage','bmi2age','ApoBstatin']
    
    trad_no_diseases = ['ApoB','Smoker','diabetes','HTN_treated','statin','bmi','bmi2','Smokersex',
             'diabetessex','HTN_treatedage','bmiage','statinage','bmi2age','ApoBstatin']

    trad_no_comor = ['ApoB','bmi','bmi2','bmiage','bmi2age']    
    
    elife_proteins = ['SeqId.4374-45','SeqId.8469-41','SeqId.4496-60','SeqId.2677-1','SeqId.7655-11','SeqId.4982-54','SeqId.2948-58']
    ho_proteins = ['SeqId.5701-81','SeqId.15422-12','SeqId.2609-59','SeqId.4374-45','SeqId.8406-17','SeqId.8469-41','SeqId.4152-58','SeqId.5400-52','SeqId.7655-11','SeqId.4125-52']
    try: 
        print('Load age dictonary')
        file = open(folder+pred_folder+"age_predict.pkl",'rb')
        age_dict = pickle.load(file)
        file.close()
        PAD2 = age_dict['{}_sexprotein_lasso'.format(dataset)][4]-X.age
        X['PAD2'] = PAD2
    except:
        print('No file to load')
    
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
            

            boruta = sorted(features_dict['{}_boruta_y{}'.format(dataset,k)])

            print('features_loaded')

        except Exception as e:
            print(e)
            print('No features')

        try: 
            file = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),'rb')
            pred_dict = pickle.load(file)
        except:
            pred_dict = {}


#         try:
#             feat = []
#             feat.extend(agesex)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesex_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(['SeqId.4374-45'])
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexGDF15_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)

#         try:
#             feat = []
#             feat.extend(['SeqId.4374-45'])
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_GDF15_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexprotein_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta_ascorr)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexprotein_ascorr_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)


#         try:
#             feat = []
# #             feat.extend(agesex)
#             feat.extend(boruta)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_protein_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)            

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(['PAD','PADage','PADsex'])
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexPAD_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)


#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(['SeqId.4374-45'])
#             feat.extend(['PAD','PADage','PADsex'])
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexGDF15PAD_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             pred_dict['{}_y{}_baseline_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)

            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             feat.extend(['PAD','PADage','PADsex'])
#             pred_dict['{}_y{}_baselinePAD_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             feat.extend(['SeqId.4374-45'])
#             feat.extend(['PAD','PADage','PADsex'])
#             pred_dict['{}_y{}_baselineGDF15PAD_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            


#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             feat.extend(['SeqId.4374-45'])
#             pred_dict['{}_y{}_baselineGDF15_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)


#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(['GDF15','GDF152','GDF15age','GDF15sex'])
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexGDF15ultra_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)


#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             feat.extend(tradblood)
#             feat.extend(['SeqId.4374-45'])
#             pred_dict['{}_y{}_baselinebloodGDF15_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            

            
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             feat.extend(boruta)
#             pred_dict['{}_y{}_baselineprotein_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
            
            
### A little bit different methods           

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexprotein_l2'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl2')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexprotein_glmnetl1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='glmnetl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            


### Baseline l2

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             pred_dict['{}_y{}_baseline_l2'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl2')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             feat.extend(tradblood)
#             pred_dict['{}_y{}_baselineblood_l2'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl2')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)


# #### Baseline2
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad2)
#             pred_dict['{}_y{}_baseline2_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad2)
#             feat.extend(boruta)
#             pred_dict['{}_y{}_baseline2protein_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad2)
#             feat.extend(['PAD','PADage','PADsex'])
#             pred_dict['{}_y{}_baseline2PAD_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad2)
#             feat.extend(['SeqId.4374-45'])
#             feat.extend(['PAD','PADage','PADsex'])
#             pred_dict['{}_y{}_baseline2GDF15PAD_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad2)
#             feat.extend(['SeqId.4374-45'])
#             pred_dict['{}_y{}_baseline2GDF15_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)   

#         try:
#             boruta.remove('SeqId.4374-45')
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad2)
#             feat.extend(boruta)
#             pred_dict['{}_y{}_baseline2proteinnoGDF15_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#             boruta.append('SeqId.4374-45')
#         except Exception as e:
#             print('Fail')
#             print(e)


#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(tradlifestyle)
#             feat.extend(['SeqId.4374-45'])
#             pred_dict['{}_y{}_lifestyleGDF15_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)   

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(tradlifestyle)
#             pred_dict['{}_y{}_lifestyle_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)   


##### No GDF15
#         boruta.remove('SeqId.4374-45')
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
#             pred_dict['{}_y{}_agesexproteinnoGDF15_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)


#         try:
#             feat = []
#             feat.extend(boruta)
#             pred_dict['{}_y{}_proteinnoGDF15_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(trad)
#             feat.extend(tradextralog)
#             feat.extend(tradcancer)
#             feat.extend(boruta)
#             pred_dict['{}_y{}_baselineproteinnoGDF15_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)


##### Different baseline for different exclusion criteria

#         try:
#             feat = []
#             feat.extend(agesex)
# #             feat.extend(trad_no_diseases)            
#             feat.extend(trad_no_comor)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_baseline2_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
# #             feat.extend(trad_no_diseases)
#             feat.extend(trad_no_comor)
#             pred_dict['{}_y{}_baseline2_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)

#         try:
#             feat = []
#             feat.extend(agesex)
# #             feat.extend(trad_no_diseases)
#             feat.extend(trad_no_comor)
#             feat.extend(boruta)
#             pred_dict['{}_y{}_baseline2protein_l1'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl1')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
            
#         try:
#             feat = []
#             feat.extend(agesex)
# #             feat.extend(trad_no_diseases)
#             feat.extend(trad_no_comor)
#             feat.extend(['SeqId.4374-45'])
#             pred_dict['{}_y{}_baseline2GDF15_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
### Papers

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(elife_proteins)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexelife_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
            
#         try:
#             feat = []
#             feat.extend(['age'])
#             feat.extend(elife_proteins)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_ageelife_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(elife_proteins)
#             feat.extend(trad2)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_baseline2elife_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(elife_proteins)
#             feat.extend(trad2)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_baseline2elife_l2'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lrl2')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(['PAD2'])
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexPAD2_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
        try:
            feat = []
            feat.extend(agesex)
            feat.extend(['PAD2'])
            feat.extend(trad2)
#             feat.extend(cumunibor15)
            pred_dict['{}_y{}_baseline2PAD2_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
            f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
            pickle.dump(pred_dict,f)
            f.close()
            print('Done k = ',k)
        except Exception as e:
            print('Fail')
            print(e)

#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(['SeqId.2652-15'])
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexPLAUR_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(['SeqId.2652-15'])
#             feat.extend(trad2)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_baseline2PLAUR_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(ho_proteins)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_agesexhopro_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(ho_proteins)
#             feat.extend(trad2)
# #             feat.extend(cumunibor15)
#             pred_dict['{}_y{}_baseline2hopro_lr'.format(dataset,k)] = pf.predict(feat=feat,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model_type='lr')
#             f = open(folder+pred_folder+"{}_{}_predict.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict,f)
#             f.close()
#             print('Done k = ',k)
#         except Exception as e:
#             print('Fail')
#             print(e)
            
            
