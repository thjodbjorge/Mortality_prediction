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
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
import sys
sys.path.append('/odinn/users/thjodbjorge/Python_functions/')
import Predict_functions as pf


# In[9]:


raw_data = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/raw_with_info.csv',index_col = 'Barcode2d' )
probe_info = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probe_info.csv', index_col = 'SeqId')

pn_info = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/pn_info_Mor/pn_info_Mor_event.csv',index_col = 'Barcode2d' )
probes_to_skip = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probes_to_skip.txt')['probe']


folder = '/odinn/users/thjodbjorge/Proteomics/Mortality2/'
feat_folder = 'Features2/'
# pred_folder = 'Predictions_cv2/'
pred_folder = 'Predictions4/'
corr_type = 'none'
skip_PCA = 1

if corr_type == 'pqtl':
    pqtl_protein = pd.read_csv('/odinn/users/egilf/pQTL/for_benedikt/pQTL_conditional_04052020.gor', sep='\t')
    # pqtl = pd.read_csv('/odinn/users/steinthora/proteomics/proteomic_project/Data/pQTL_Merged_08052020.csv', sep = '\t', index_col = 'PN')
    pqtl = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/pqtl/pqtl_combined_meanimp.csv',index_col = 'PN')

    pqtl = pd.merge(pn_info['PN'],pqtl,left_on='PN',right_index=True)
    pqtl.drop('PN',axis=1,inplace=True)
    pro_pqtl = {}
    for i in raw_data.iloc[:,16:].columns:
        pro_pqtl[i] = list(pqtl_protein[pqtl_protein.SeqId == i[6:].replace('-','_')]['SentinelMarker'])
        
        
        
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
keep_samples_keys = ['Old_18105','Old_60105']#,'Old_18105']#,'Old_18105','Old_6080']#,'Old_60105','Old_6080']
# K = [9]
K = [9,4,1,14]
# K =[14]
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
    
    batch_var = pd.get_dummies(pn_info['batch'],drop_first = True).columns
    X[batch_var] = pd.get_dummies(pn_info['batch'],drop_first = True)
    
    
    PRS = ['nonHDL_prs', 'HT_prs', 'CAD_prs', 'Cancer_prs', 'Stroke2_prs', 'alz_Jansen',
       'pgc_adhd_2017', 'PD_Nalls_2018', 'edu_160125', 'dep_2018', 'bpd_2018',
       'giant_bmi', 'schizo_clozuk', 'iq_2018', 'ipsych_pgc_aut_2017',
       'pgc_Anorexia_2019']
    X[PRS] = pn_info[PRS]

    for prs in PRS:
        no_prs = (X[prs].isna()); print(no_prs.sum())
        no_prs_ind = X[no_prs].index
        X.loc[no_prs_ind,prs] = X.loc[I_train,prs].mean()
    X.loc[no_prs_ind,PRS]
    
    X['site'] = (pn_info['site'] == 'DC').astype(int)
    X['Sample_age'] = pn_info['Sample_age']
    
    trad = ['ApoB','Smoker','diabetes','HTN_treated','statin','CAD','MI','bmi','bmi2']
    tradage = ['ApoBage','Smokerage','diabetesage','CADage','MIage','HTN_treatedage','bmiage']
    tradsex = ['ApoBsex','Smokersex','diabetessex','CADsex','MIsex','HTN_treatedsex','bmisex']
    
    tradextralog = ['Smokersex','diabetessex','CADsex','CADage','MIage','HTN_treatedage','bmiage','statinage','bmi2age','ApoBstatin']
#     tradextralognosex = ['CADage','MIage','HTN_treatedage','bmiage']
    tradblood = ['Creatinine','Triglycerides','Platelets','Platelets2','Plateletsage','Creatinineage','Platelets2age']
    tradcancer = ['cancer','cancerage']    
#     extra_cancer = ['cancer1y','cancer5y','cancer1yage','cancer5yage']

    trad2 = ['ApoB','Smoker','diabetes','HTN_treated','statin','CAD','MI','bmi','bmi2','Stroke','Smokersex',
             'diabetessex','CADsex','CADage','MIage','HTN_treatedage','bmiage','statinage','bmi2age','ApoBstatin',
            'cancer','cancerage','cancersex']
    
    
    if corr_type == 'pqtl':
        X =X.merge(pqtl,how = 'left', right_index=True, left_index=True)
    
    X_train = X.loc[I_train]
    X_test = X.loc[I_test]
    
    if corr_type == 'pqtl':
        ### Correct for pqtls
        for m in pqtl.columns:
            no_p = (X[m].isna());# print(no_p.sum())
            no_p_ind = X[no_p].index
            X_train.loc[I_train.intersection(no_p_ind),m] = X_train[m].mean()
            X_test.loc[I_test.intersection(no_p_ind),m] = X_test[m].mean()


        for p in all_protein:    
            pqtl_model = sm.OLS(X_train[p],sm.add_constant(X_train[pro_pqtl[p]])).fit()
            corr_train = pqtl_model.predict(sm.add_constant(X_train[pro_pqtl[p]]))
            corr_test = pqtl_model.predict(sm.add_constant(X_test[pro_pqtl[p]]))
        #     corr_train.columns = all_protein
            X_train[p] = X_train[p] - corr_train
            X_test[p] = X_test[p] - corr_test
    
    
    if corr_type == 'PCA':
        pca1 = PCA(skip_PC)
        x_pca1 = pca1.fit_transform(X_train[all_protein])
        x_1 = pca1.inverse_transform(x_pca1)
        X_train[all_protein] = X_train[all_protein] - x_1

        x_pca1 = pca1.transform(X_test[all_protein])
        x_1 = pca1.inverse_transform(x_pca1)
        X_test[all_protein] = X_test[all_protein] - x_1
        
    if corr_type == 'batch':
        for p in all_protein:    
            corr_model = sm.OLS(X_train[p],sm.add_constant(X_train[batch_var])).fit()
            corr_train = corr_model.predict(sm.add_constant(X_train[batch_var]))
            corr_test = corr_model.predict(sm.add_constant(X_test[batch_var]))
        #     corr_train.columns = all_protein
            X_train[p] = X_train[p] - corr_train
            X_test[p] = X_test[p] - corr_test       
            
            
    if corr_type == 'sitesampleage':
        print('Correct proteins fot site and sample age')
        for p in all_protein:    
            corr_model = sm.OLS(X_train[p],sm.add_constant(X_train[['site','Sample_age']])).fit()
            corr_train = corr_model.predict(sm.add_constant(X_train[['site','Sample_age']]))
            corr_test = corr_model.predict(sm.add_constant(X_test[['site','Sample_age']]))
        #     corr_train.columns = all_protein
            X_train[p] = X_train[p] - corr_train
            X_test[p] = X_test[p] - corr_test
        print('Correction done')     
        
        
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
            boruta15 = sorted(features_dict['{}_boruta_y14'.format(dataset)])
        

            
            
            print('features_loaded')

        except Exception as e:
            print(e)
            print('No features')

        try: 
#             file = open(folder+pred_folder+"{}_{}_predict_cv_XGB.pkl".format(endpoint,dataset),'rb')
#             file = open(folder+pred_folder+"{}_{}_predict_cv_MLP.pkl".format(endpoint,dataset),'rb')
#             file = open(folder+pred_folder+"{}_{}_predict_cv_SVM.pkl".format(endpoint,dataset),'rb')
#             file = open(folder+pred_folder+"{}_{}_predict_cv_corr.pkl".format(endpoint,dataset),'rb')
            file = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),'rb')
            pred_dict_cv = pickle.load(file)
        except:
            print('No file to load')
            pred_dict_cv = {}
  
     
#         try:
#             feat = []
#             feat.extend(agesex)
#             out = pf.predict_surv_cv2(feat=feat,kf=kf,X=X_train,y_surv = ysurv_train,y=y[k][I_train], k = k,
#                                       event_col = 'event', time_to_event_col = 'time_to_event',
#                                       model_type = 'cox',feat_sel_type = None)
#             pred_dict_cv['{}_agesex_cox_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')      
    
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'lr',feat_sel_type = None)
#             pred_dict_cv['{}_agesex_lr_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')

  
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
#             out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'lrl1',feat_sel_type = None)
#             pred_dict_cv['{}_agesexboruta_l1_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')    
  

#         try:
#             feat = []
# #             feat.extend(agesex)
#             feat.extend(boruta)
#             out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'lrl2',feat_sel_type = None)
#             pred_dict_cv['{}_boruta_l2_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')    
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
#             out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'lrl2',feat_sel_type = None)
#             pred_dict_cv['{}_agesexboruta_l2_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')    
             
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
#             out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'lrl1l2',feat_sel_type = None)
#             pred_dict_cv['{}_agesexboruta_elnet_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')      

            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
#             out = pf.predict_surv_cv2(feat=feat,kf=kf,X=X_train,y_surv = ysurv_train,y=y[k][I_train], k = k,
#                                       event_col = 'event', time_to_event_col = 'time_to_event',
#                                       model_type = 'coxl1',feat_sel_type = None)
#             pred_dict_cv['{}_agesexboruta_coxl1_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')   
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
#             out = pf.predict_surv_cv2(feat=feat,kf=kf,X=X_train,y_surv = ysurv_train,y=y[k][I_train], k = k,
#                                       event_col = 'event', time_to_event_col = 'time_to_event',
#                                       model_type = 'coxl2',feat_sel_type = None)
#             pred_dict_cv['{}_agesexboruta_coxl2_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')       
          
            
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
#             out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'xgb',feat_sel_type = None)
#             pred_dict_cv['{}_agesexboruta_xgb_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv_XGB.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')                  
            
            
#         try:
#             feat = []
#             feat.extend(agesex)
#             feat.extend(boruta)
#             out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'mlp',feat_sel_type = None)
#             pred_dict_cv['{}_agesexboruta_mlp_y{}'.format(dataset,k)] = out

#             f = open(folder+pred_folder+"{}_{}_predict_cv_MLP.pkl".format(endpoint,dataset),"wb")
#             pickle.dump(pred_dict_cv,f)
#             f.close()
#         except Exception as e:
#             print(e)
#             print('Fail')    


        try:
            feat = []
            feat.extend(agesex)
#             feat.extend(agebins)
            feat.extend(trad2)
            out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'lr',feat_sel_type = None)
            pred_dict_cv['{}_baseline2_lr_y{}'.format(dataset,k)] = out

            f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
            pickle.dump(pred_dict_cv,f)
            f.close()
        except Exception as e:
            print(e)
            print('Fail')   

# # # ### PRS lr


        try:
            feat = []
            feat.extend(agesex)
#             feat.extend(agebins)
            feat.extend(trad2)
            feat.extend(PRS)
            out = pf.predict_cv(feat=feat,kf=kf,X=X_train,y=y[k][I_train],model_type = 'lr',feat_sel_type = None)
            pred_dict_cv['{}_baseline2prs_lr_y{}'.format(dataset,k)] = out

            f = open(folder+pred_folder+"{}_{}_predict_cv.pkl".format(endpoint,dataset),"wb")
            pickle.dump(pred_dict_cv,f)
            f.close()
        except Exception as e:
            print(e)
            print('Fail')   
  
