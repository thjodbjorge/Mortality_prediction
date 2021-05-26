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
probes_to_skip = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probes_to_skip.txt')['probe'] # this originally cam from sigrún
nopro = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/no_protein_probes.txt', header = None)[0] # non-priten probes that were included 
probes_to_skip = set(probes_to_skip).union(set(nopro))


folder = '/odinn/users/thjodbjorge/Proteomics/Mortality2/'
feat_folder = 'Features2/'

endpoints = ['death']
# event_date = event_date_death
time_to_event = pn_info.time_to_death
no_event_before = pn_info.no_death_before
# endpoints = ['Neoplasm','Nervous','Circulatory','Respiratory','Other']
for endpoint in endpoints:
    if endpoint == 'death':
        use_event = pn_info.event_death
    elif endpoint == 'Neoplasm':
        use_event = pn_info.event_death & (pn_info.Cause_of_death == 'Neoplasm')
    elif endpoint == 'Nervous':
        use_event = pn_info.event_death & (pn_info.Cause_of_death == 'Nervous')
    elif endpoint == 'Circulatory':
        use_event = pn_info.event_death & (pn_info.Cause_of_death == 'Circulatory')
    elif endpoint == 'Respiratory':
        use_event = pn_info.event_death & (pn_info.Cause_of_death == 'Respiratory')
    elif endpoint == 'Other':
        use_event = pn_info.event_death & (pn_info.Cause_of_death == 'Other')

    y = []
    for i in range(1,19):
        y.append(use_event & (time_to_event <= i))

    # kf = KFold(n_splits=10, random_state=10, shuffle=False) 
    I_train_main, I_test_main = train_test_split(pn_info.index, train_size=0.7, random_state = 10)
    I_val_main, I_test_main = train_test_split(I_test_main, train_size=0.5, random_state = 10)


    file = open(folder+"{}_keep_samples.pkl".format('Mor'),'rb')
    keep_samples_dict = pickle.load(file)

    print(keep_samples_dict.keys())
#     keep_samples_keys = ['Old_60105','Old_18105','Old_6080','Old_18105_Neoplasms','Old_18105_I','Old_18105_J','Old_18105_G','Old_18105_Other']
#     keep_samples_keys = ['Old_60105_AC2','Old_18105_AC2']
    keep_samples_keys = ['Old_18105']
#     keep_samples_keys = ['Old_no_diseases_18105','Old_no_diseases_60105','Old_no_comor_18105','Old_no_comor_60105']
    for dataset in keep_samples_keys:
        print(dataset)
        keep_samples = keep_samples_dict[dataset]

        I_train = I_train_main.intersection(keep_samples)#.intersection(have_prs)
        I_test = I_val_main.intersection(keep_samples)#.intersection(have_prs)

        print('Training set: {}, MI within 15: {}, 10: {}, 5: {}, 2: {}'.format(len(I_train),y[14][I_train].sum(),y[9][I_train].sum(),y[4][I_train].sum(),y[1][I_train].sum()))
#         print('Test set: {}, MI within 15: {}, 10: {}, 5: {}, 2: {}'.format(len(I_test),y[14][I_test].sum(),y[9][I_test].sum(),y[4][I_test].sum(),y[1][I_test].sum()))

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

        X['ApoB'] = X['SeqId.2797-56']
        X['Smoker'] = pn_info['Smoker'].astype(int).values
        X['diabetes'] = pn_info['T2D'].astype(int).values
        X['HTN_treated'] = pn_info[['HTN_treated']].astype(int).values
        X['statin'] = pn_info['statin'].astype(int).values
#         X['statin'] = pn_info['statin_estimate_unsure'].astype(int).values
        X['ageHTN_treated'] =  X['age']*X['HTN_treated']    
        
        X['bmi'] = pn_info['bmi']

        no_bmi = (X['bmi'].isna())
        no_bmi_ind = X[no_bmi].index
        X.loc[no_bmi_ind,'bmi'] = X.loc[I_train].bmi.mean()

        X['bmiage'] = X['bmi']*X['age']
        X['bmisex'] = X['bmi']*X['sex']    
    
        X['CAD'] = ~pn_info.no_CAD_before
        X['MI'] = ~pn_info.no_MI_before
        
        X['site'] = (pn_info['site'] == 'DC').astype(int)
        X['Sample_age'] = pn_info['Sample_age']      

        batch_var = pd.get_dummies(pn_info['batch'],drop_first = True).columns
        X[batch_var] = pd.get_dummies(pn_info['batch'],drop_first = True)

        X_train = X.loc[I_train]
#         X_test = X.loc[I_test]

        train_mean = X_train.mean()
        train_std = X_train.std()

        X_train = (X_train-train_mean)/train_std
#         X_test = (X_test-train_mean)/train_std

        ## For survival analysis    
        X_train['event'] = use_event[I_train]
#         X_test['event'] = use_event[I_test]

        tte_train = time_to_event[I_train]
#         tte_test = time_to_event[I_test]

    #     ysurv_train = pd.DataFrame()
    #     ysurv_train['event'] = use_event[I_train]
    #     ysurv_train['time_to_event'] = time_to_event[I_train]

#         K =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#         K =[4,9,14,5,6,7,8,10,11,12,13]
        K = [0,1,2,3]
#         K = [4,9,14]
        try:
#             file = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),'rb')
#             features_dict = pickle.load(file)
            file = open(folder+feat_folder+"{}_{}_features_tmp.pkl".format(endpoint,dataset),'rb')
            features_dict = pickle.load(file)
        except:
            features_dict = {}
    #     K = [14]
#         K = [4]
        for k in K:

            y_train = y[k][I_train]
#             y_test = y[k][I_test]
            ## Simple bortua
            if 1:
                print('Boruta feature selection')
                np.random.seed(10)
                forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
                feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
                feat_selector.fit(X_train[all_protein].values, y_train.values)
                bor_feat = X_train[all_protein].loc[:,feat_selector.support_].columns.tolist()
                bor_feat_weak =  X_train[all_protein].loc[:,feat_selector.support_weak_].columns.tolist()

                features_dict['{}_boruta_y{}'.format(dataset,k)] = bor_feat
                features_dict['{}_boruta_weak_y{}'.format(dataset,k)] = bor_feat_weak

                print(len(bor_feat))
                f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
                pickle.dump(features_dict,f)
                f.close()
                                
                
            if 0:
                print('Boruta feature selection age + sex corrected proteins')
                corr_col = agesex
                corr_pred = sm.OLS(X_train[all_protein],sm.add_constant(X_train[corr_col])).fit().predict(sm.add_constant(X_train[corr_col]))
                corr_pred.columns = all_protein
                corr_pro = X_train[all_protein] - corr_pred
                corr_pro = corr_pro/corr_pro.std()
                
                
                np.random.seed(10)
                forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
                feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
                feat_selector.fit(corr_pro.values, y_train.values)
                
                bor_feat = corr_pro.loc[:,feat_selector.support_].columns.tolist()
                bor_feat_weak =  corr_pro.loc[:,feat_selector.support_weak_].columns.tolist()

                features_dict['{}_boruta_agesexcorr_y{}'.format(dataset,k)] = bor_feat
                features_dict['{}_boruta_weak_agesexcorr_y{}'.format(dataset,k)] = bor_feat_weak

                print(len(bor_feat))
                f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
                pickle.dump(features_dict,f)
                f.close()
                
                
            if 0:
                print('Boruta feature selection batch and site corrected proteins')
                corr_col = ['site']
                corr_col.extend(batch_var)
                corr_pred = sm.OLS(X_train[all_protein],sm.add_constant(X_train[corr_col])).fit().predict(sm.add_constant(X_train[corr_col]))
                corr_pred.columns = all_protein
                corr_pro = X_train[all_protein] - corr_pred
                corr_pro = corr_pro/corr_pro.std()
                
                
                np.random.seed(10)
                forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
                feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
                feat_selector.fit(corr_pro.values, y_train.values)
                
                bor_feat = corr_pro.loc[:,feat_selector.support_].columns.tolist()
                bor_feat_weak =  corr_pro.loc[:,feat_selector.support_weak_].columns.tolist()

                features_dict['{}_boruta_batchsitecorr_y{}'.format(dataset,k)] = bor_feat
                features_dict['{}_boruta_weak_batchsitecorr_y{}'.format(dataset,k)] = bor_feat_weak

                print(len(bor_feat))
                f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
                pickle.dump(features_dict,f)
                f.close()
                
            if 0:
                print('Boruta feature selection age + sex corrected proteins +  age and sex features')
                corr_col = agesex
                corr_pred = sm.OLS(X_train[all_protein],sm.add_constant(X_train[corr_col])).fit().predict(sm.add_constant(X_train[corr_col]))
                corr_pred.columns = all_protein
                corr_pro = X_train[all_protein] - corr_pred
                corr_pro = corr_pro/corr_pro.std()
                corr_pro[agesex] = X_train[agesex]
                
                np.random.seed(10)
                forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
                feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
                feat_selector.fit(corr_pro.values, y_train.values)
                
                bor_feat = corr_pro.loc[:,feat_selector.support_].columns.tolist()
                bor_feat_weak =  corr_pro.loc[:,feat_selector.support_weak_].columns.tolist()

                features_dict['{}_boruta_agesexcorr_asfeat_y{}'.format(dataset,k)] = bor_feat
                features_dict['{}_boruta_weak_agesexcorr_asfeat_y{}'.format(dataset,k)] = bor_feat_weak

                print(len(bor_feat))
                f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
                pickle.dump(features_dict,f)
                f.close()
                

            ## Botua on univariate cox significant
            if 0:
                file = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),'rb')
                uni_dict = pickle.load(file)
                univariate =  uni_dict['{}_uni_cox_005_y4'.format(dataset)]
                
                print('Boruta feature selection')
                np.random.seed(10)
                forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
                feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
                feat_selector.fit(X_train[univariate].values, y_train.values)
                bor_feat = X_train[univariate].loc[:,feat_selector.support_].columns.tolist()
                bor_feat_weak =  X_train[univariate].loc[:,feat_selector.support_weak_].columns.tolist()

                features_dict['{}_uniboruta_y{}'.format(dataset,k)] = bor_feat
                features_dict['{}_uniboruta_weak_y{}'.format(dataset,k)] = bor_feat_weak

                print(len(bor_feat))

                f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
                pickle.dump(features_dict,f)
                f.close()

            ## Anova
            if 0:
                try:
                    feat_selector = SelectFdr(f_classif,alpha=0.05)
                    feat_selector.fit(X_train[all_protein], y_train)
                    features_dict['{}_FDRanova_y{}'.format(dataset,k)] = X_train[all_protein].loc[:,feat_selector.get_support()].columns

                    f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
                    pickle.dump(features_dict,f)
                    f.close()
                except:
                    print('Fanova fail')
                    
            ## 10 runs of boruta
            if 0:        
                np.random.seed(10)
                sel_feat = []
                for i in range(10):
                    forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
                    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
                    feat_selector.fit(X_train[all_protein].values, y[k][I_train].values)
                    sel_feat.append(X_train[all_protein].loc[:,feat_selector.support_].columns.tolist())

                sel_feat_uni = set()
                for x in sel_feat:
                    sel_feat_uni = sel_feat_uni.union(x)
                print(len(list(sel_feat_uni)))
                sel_feat_int = set(sel_feat[0])
                for x in sel_feat:
                    sel_feat_int = sel_feat_int.intersection(x)
                print(len(sel_feat_int))
                features_dict['{}_bor10uni_y{}'.format(dataset,k)] = list(sel_feat_uni)
                features_dict['{}_bor10int_y{}'.format(dataset,k)] = list(sel_feat_int)
                # features_dict['{}_bor10ally{}'.format(dataset,k)] = sel_feat

                print(sel_feat_int)
                print(sel_feat_uni)

                # In[ ]:

                f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
                pickle.dump(features_dict,f)
                f.close()     
            ## Recursive CV on boruta set
            if 0:    
                use = []
                use.extend(agesex)
                use.extend(features_dict['{}_boruta_y{}'.format(dataset,k)])

                model = LogisticRegression(penalty='none',n_jobs=-1, solver ='saga', max_iter=2000, random_state = 10)
                feat_selector = RFECV(model, step=1, cv=5, scoring = 'neg_log_loss')    
                feat_selector.fit(X_train[use], y_train)

                features_dict['{}_borutaRFECV_y{}'.format(dataset,k)] = X_train[use[len(agesex):]].loc[:,feat_selector.support_[len(agesex):]].columns.tolist()

                f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
                pickle.dump(features_dict,f)
                f.close()
                           

                                        
        if 0:
                ## protein
            all_set = set(features_dict['{}_uniboruta_y0'.format(dataset)])
            features_dict['{}_cumuniboruta_y0'.format(dataset)] = all_set
            sets = [all_set]
            for k in range(15):
                all_set = all_set.union(set(features_dict['{}_uniboruta_y{}'.format(dataset,k)]))
                features_dict['{}_cumuniboruta_y{}'.format(dataset, k)] = all_set
                print(len(all_set))
                sets.append(all_set)

            f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
            pickle.dump(features_dict,f)
            f.close()
        if 0:
            all_set = set(features_dict['{}_boruta_y0'.format(dataset)])
            features_dict['{}_cumboruta_y0'.format(dataset)] = all_set
            sets = [all_set]
            for k in range(15):
                all_set = all_set.union(set(features_dict['{}_boruta_y{}'.format(dataset,k)]))
                features_dict['{}_cumboruta_y{}'.format(dataset, k)] = all_set
                print(len(all_set))
                sets.append(all_set)

            f = open(folder+feat_folder+"{}_{}_features.pkl".format(endpoint,dataset),"wb")
            pickle.dump(features_dict,f)
            f.close()
            
            
