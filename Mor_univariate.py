#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix,log_loss
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, quantile_transform
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, RFECV, SelectPercentile, SelectFpr, SelectFdr, SelectFwe
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt
import pickle

from scipy.stats import t,ttest_ind,sem, pearsonr, spearmanr
import seaborn as sns
import lifelines as ll
from lifelines.utils.sklearn_adapter import sklearn_adapter
import sys
sys.path.append('/odinn/users/thjodbjorge/Python_functions/')
import Predict_functions as pf
import warnings
# warnings.filterwarnings('error')

raw_data = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/raw_with_info.csv',index_col = 'Barcode2d' )
probe_info = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probe_info.csv', index_col = 'SeqId')

pn_info = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/pn_info_Mor/pn_info_Mor_event.csv',index_col = 'Barcode2d' )
probes_to_skip = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/probes_to_skip.txt')['probe']
nopro = pd.read_csv('/odinn/users/thjodbjorge/Proteomics/Data/no_protein_probes.txt', header = None)[0] # non-protein probes that were included 
probes_to_skip = set(probes_to_skip).union(set(nopro))


folder = '/odinn/users/thjodbjorge/Proteomics/Mortality2/'

# endpoints = ['death']
# endpoints = ['death','Cdeath','Gdeath','Ideath','Jdeath','Otherdeath']
# event_date = event_date_death
time_to_event = pn_info.time_to_death
no_event_before = pn_info.no_death_before
# endpoints = ['Neoplasm','Nervous','Circulatory','Respiratory','Other','death']
# endpoints = ['Neoplasm']
# endpoints = ['Circulatory']
# endpoints = ['Respiratory']
# endpoints = ['Other']
endpoints = ['death']
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


I_train_main, I_test_main = train_test_split(pn_info.index, train_size=0.7, random_state = 10)

file = open(folder+"{}_keep_samples.pkl".format('Mor'),'rb')
keep_samples_dict = pickle.load(file)
file.close()
# 'Old_60105','Old_18105',


## Select model typa and dataset
model_type = 'LR'
use_all_data=False
use_qt_transform = False
agesex_corr = True
site_corr= False
batch_corr = False
covar_agesex = False
batchsite_corr = False
keep_samples_keys = ['Old_18105_Neoplasms','Old_18105_I','Old_18105_J','Old_18105_G','Old_18105_Other','Old_18105','Old_60105','Old_6080']
# keep_samples_keys = ['Old_60105_AC2','Old_18105_AC2']
# keep_samples_keys = ['Old_18105_Other']
# keep_samples_keys = ['Old_18105_I']
# keep_samples_keys = ['Old_18105_J']
# keep_samples_keys = ['Old_18105_G']
# keep_samples_keys = ['Old_UVS_18105','Old_DC_18105']
# keep_samples_keys = ['Old_women_18105','Old_women_60105','Old_women_6080']
# keep_samples_keys = ['Old_18105','Old_60105','Old_6080']
# keep_samples_keys = ['Old_6570','Old_1860','Old_6070','Old_7080','Old_80105','Old_8090','Old_90105','Old_18105','Old_60105','Old_6080']
k=4 # Five year univariate
# k = 9
for dataset in keep_samples_keys:
# for dataset in keep_samples_dict:
    print(dataset)
    keep_samples = keep_samples_dict[dataset]
    
    if use_all_data:
        I_train = keep_samples.copy()
    else:
        I_train = I_train_main.intersection(keep_samples)#.intersection(have_prs)

    print('Training set: {}, Death within 15: {}, 10: {}, 5: {}, 2: {}'.format(len(I_train),y[14][I_train].sum(),y[9][I_train].sum(),y[4][I_train].sum(),y[1][I_train].sum()))
#     print('Test set: {}, MI within 15: {}, 10: {}, 5: {}, 2: {}'.format(len(I_test),y[14][I_test].sum(),y[9][I_test].sum(),y[4][I_test].sum(),y[1][I_test].sum()))

    ### Select data and normalize

    X = np.log(raw_data.iloc[:,16:].drop(probes_to_skip,axis=1))
    allp = X.columns
    
    X['sex'] = pn_info[['sex']].values-1
    X['age'] = pn_info[['Age_at_sample_collection_2']].values
    X['age2'] = X['age']**2
    X['age3'] = X['age']**3
    X['agesex'] = X['age']*X['sex']
    X['age2sex'] = X['age2']*X['sex']
    agesex = ['age','sex','agesex','age2','age2sex']
    X['agebin'] = pn_info['agebin']
    
    X['site'] = (pn_info['site'] == 'DC').astype(int)
    X['Sample_age'] = pn_info['Sample_age']      

    batch_var = pd.get_dummies(pn_info['batch'],drop_first = True).columns
    X[batch_var] = pd.get_dummies(pn_info['batch'],drop_first = True)
    
    
    X_train = X.loc[I_train]
    
    if use_qt_transform:
        print('Qt transform')
        X_train[allp] = pd.DataFrame(quantile_transform(X_train[allp], n_quantiles=50000, output_distribution = 'normal', copy =False),
                         index=X_train.index,columns=allp)
    if agesex_corr:
#         X_train[allp] =             
        corr_model = sm.OLS(X_train[allp],sm.add_constant(X_train[agesex])).fit()
        corr_train = corr_model.predict(sm.add_constant(X_train[agesex]))
        corr_train.columns = allp
        X_train[allp] = X_train[allp] - corr_train

    if site_corr:
        print('Correct proteins for site and sample age')
        for p in allp:    
            corr_model = sm.OLS(X_train[p],sm.add_constant(X_train[['site','Sample_age']])).fit()
            corr_train = corr_model.predict(sm.add_constant(X_train[['site','Sample_age']]))
        #     corr_train.columns = all_protein
            X_train[p] = X_train[p] - corr_train
            
            
    if batch_corr:
        print('Correct proteins for batch')
        for p in allp:    
            corr_model = sm.OLS(X_train[p],sm.add_constant(X_train[batch_var])).fit()
            corr_train = corr_model.predict(sm.add_constant(X_train[batch_var]))
        #     corr_train.columns = all_protein
            X_train[p] = X_train[p] - corr_train            
            
    if batchsite_corr:
        print('Correct proteins for batch')
        for p in allp:    
            corr_feat = ['site']
            corr_feat.extend(batch_var)
            corr_model = sm.OLS(X_train[p],sm.add_constant(X_train[corr_feat])).fit()
            corr_train = corr_model.predict(sm.add_constant(X_train[corr_feat]))
        #     corr_train.columns = all_protein
            X_train[p] = X_train[p] - corr_train    
    
        
    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train = (X_train-train_mean)/train_std

    y_train = y[k][I_train]
    
    X_train['event'] = use_event[I_train]
    
    tte_train = time_to_event[I_train]
  
    linear_pvals = []
    linear_betas = []
    log_pvals = []
    log_betas = []
    cox_coefs = []
    cox_pvals = []
    cox_ph = []
    cox_ph_covar = []
    AFT_coefs = []
    AFT_pvals = []
    fail_count = 0
    cox_fail_count = 0
    AFT_fail_count = 0
    for i in range(len(allp)):
        pro = X_train.iloc[:,i]
        pro = (pro-pro.mean())/pro.std()
        
        if model_type == 'LR':
            datas = X_train.loc[:,['age','sex','agesex','age2','age2sex']]
    #         datas = X_train[['age','age2']]
            datas['Mor'] = y_train*1
            datas[pro.name] = pro
    #         model = sm.OLS(pro,sm.add_constant(datas.drop(pro.name,axis=1)))
    #         res = model.fit(disp=0)
    #         linear_pvals.append(res.pvalues[-1])
    #         linear_betas.append(res.params[-1])
    #         print(res.params, res.pvalues)
            try:
                model = sm.Logit(datas['Mor'],sm.add_constant(datas.drop('Mor',axis=1)))
                res = model.fit(disp=0)
                log_pvals.append(res.pvalues[-1])
                log_betas.append(res.params[-1])
            except:
                fail_count += 1
                print('Fail: ', fail_count)
                log_pvals.append(np.nan)
                log_betas.append(np.nan)
                
        elif model_type == 'LRnoage':
            datas = X_train.loc[:,['sex']]
    #         datas = X_train[['age','age2']]
            datas['Mor'] = y_train*1
            datas[pro.name] = pro
    #         model = sm.OLS(pro,sm.add_constant(datas.drop(pro.name,axis=1)))
    #         res = model.fit(disp=0)
    #         linear_pvals.append(res.pvalues[-1])
    #         linear_betas.append(res.params[-1])
    #         print(res.params, res.pvalues)
            try:
                model = sm.Logit(datas['Mor'],sm.add_constant(datas.drop('Mor',axis=1)))
                res = model.fit(disp=0)
                log_pvals.append(res.pvalues[-1])
                log_betas.append(res.params[-1])
            except:
                fail_count += 1
                print('Fail: ', fail_count)
                log_pvals.append(np.nan)
                log_betas.append(np.nan)
                
                
        elif model_type == 'LRbss':
            feat = ['age','sex','agesex','age2','age2sex']
            feat.extend(batch_var)
            feat.extend(['site','Sample_age'])
            datas = X_train.loc[:,feat]
            datas['Mor'] = y_train*1
            datas[pro.name] = pro
            try:
                model = sm.Logit(datas['Mor'],sm.add_constant(datas.drop('Mor',axis=1)))
                res = model.fit(disp=0,method = 'lbfgs')
                log_pvals.append(res.pvalues[-1])
                log_betas.append(res.params[-1])
            except:
                fail_count += 1
                print('Fail: ', fail_count)
                log_pvals.append(np.nan)
                log_betas.append(np.nan)
        elif model_type == 'LRss':
            feat = ['age','sex','agesex','age2','age2sex']
#             feat.extend(batch_var)
            feat.extend(['site','Sample_age'])
            datas = X_train.loc[:,feat]
            datas['Mor'] = y_train*1
            datas[pro.name] = pro
            try:
                model = sm.Logit(datas['Mor'],sm.add_constant(datas.drop('Mor',axis=1)))
                res = model.fit(disp=0)
                log_pvals.append(res.pvalues[-1])
                log_betas.append(res.params[-1])
            except:
                fail_count += 1
                print('Fail: ', fail_count)
                log_pvals.append(np.nan)
                log_betas.append(np.nan)                

            
        elif model_type == 'cox':
    #         datas = X_train[['age','sex','agesex','age2','age2sex','event']]
#             datas = X_train.loc[:,['age','age2','sex','event']]      
            if covar_agesex:
                datas = X_train.loc[:,['age','sex','event']]  
            else:
#                 datas = X_train[['age','sex','agesex','age2','age2sex','event']]
                datas = X_train.loc[:,['event']]  
            datas[pro.name] = pro
            datas['tte'] = tte_train
            try:
                model = ll.CoxPHFitter()
                print(i, end = " ")
                model.fit(datas,duration_col = 'tte',event_col = 'event')#, step_size = 0.1,)
                cox_coefs.append(model.summary.loc[pro.name, 'coef'])
                cox_pvals.append(model.summary.loc[pro.name, 'p'])
            except Exception as e:
                cox_fail_count += 1
                print('\n',pro.name ,' Fail: ',cox_fail_count)
                print(e)
                cox_coefs.append(np.nan)
                cox_pvals.append(np.nan)        
        elif model_type == 'coxstrat':
            datas = X_train.loc[:,['age','sex','agesex','age2','age2sex','agebin','event']]
  
            datas[pro.name] = pro
            datas['tte'] = tte_train
            try:
                model = ll.CoxPHFitter()
                print(i, end = " ")
                model.fit(datas,strata = 'agebin',duration_col = 'tte',event_col = 'event')#, step_size = 0.01,)
                cox_coefs.append(model.summary.loc[pro.name, 'coef'])
                cox_pvals.append(model.summary.loc[pro.name, 'p'])
                out = ll.statistics.proportional_hazard_test(model, datas, time_transform='km')
                cox_ph.append(out.p_value[-1])
                cox_ph_covar.append(np.min(out.p_value[:-1]))
            except Exception as e:
                cox_fail_count += 1
                print('\n',pro.name ,' Fail: ',cox_fail_count)
                print(e)
                cox_coefs.append(np.nan)
                cox_pvals.append(np.nan)            
        elif model_type == 'AFT':  
            datas = X_train.loc[:,['age','sex','agesex','age2','age2sex','event']]
#             datas = X_train[['age','age2','sex','event']]        
            datas[pro.name] = pro
            datas['tte'] = tte_train
            try:                
                model = ll.LogLogisticAFTFitter()
                model.fit(datas,duration_col = 'tte',event_col = 'event',fit_intercept = True)
                print(i)
                AFT_coefs.append(model.summary.loc['alpha_','coef'].loc[pro.name])
                AFT_pvals.append(model.summary.loc['alpha_','p'].loc[pro.name])
            except Exception as e:
                print(e)
                AFT_fail_count += 1
                print('Fail: ', AFT_fail_count)
                AFT_coefs.append(np.nan)
                AFT_pvals.append(np.nan)     
        else:
            print('Unknown model type')
#     result_df = pd.DataFrame([linear_betas,linear_pvals,log_betas,log_pvals,cox_coefs,cox_pvals],index=['linear_beta','linear_pval','log_beta','log_pval', 'cox_coef','cox_pval'],columns=allp)
#     result_df = pd.DataFrame([log_betas,log_pvals,cox_coefs,cox_pvals],index=['log_beta','log_pval', 'cox_coef','cox_pval'],columns=allp)
    if (model_type == 'LR') | (model_type == 'LRbss') | (model_type == 'LRb') | (model_type == 'LRss')|(model_type == 'LRnoage'):
        result_df = pd.DataFrame([log_betas,log_pvals],index=['log_beta','log_pval'],columns=allp)
    elif model_type == 'cox':
        result_df = pd.DataFrame([cox_coefs,cox_pvals],index=['cox_coef','cox_pval'],columns=allp)
    elif model_type == 'AFT':
        result_df = pd.DataFrame([AFT_coefs,AFT_pvals],index=['AFT_coef','AFT_pval'],columns=allp)
    elif model_type == 'coxstrat':
        result_df = pd.DataFrame([cox_coefs,cox_pvals,cox_ph,cox_ph_covar],index=['cox_coef','cox_pval','ph_assumption_protein','ph_assumption_covar'],columns=allp)
    else:
        print('Unknown modeltype in create dataframe')

    res = result_df
    ind = []
    for string in res.columns:
        ind.append(string.replace('SeqId.',''))
    ind = pd.DataFrame(ind,index = res.columns, columns = ['new_ind'])
    # probe_info.loc[ind[0:5],['Target','TargetFullName']]
    rest = res.transpose()
    rest =  rest.join(ind)
    rest.set_index('new_ind', inplace = True)
    rest = rest.join(probe_info[['Target','TargetFullName','UniProt','GeneName']])
    
    if use_all_data:
        dataset = dataset+'_all'
    if use_qt_transform:
        dataset = dataset+'_qt'
    if agesex_corr:
        dataset= dataset+'_ascorr'
    if site_corr:
        dataset = dataset+'_sitecorr'
    if batch_corr:
        dataset = dataset+'_batchcorr'
    if batchsite_corr:
        dataset = dataset+'_batchsitecorr'
    
    if covar_agesex:
        dataset = dataset+'_as'
    
    
    if model_type == 'LR':
        rest.to_csv(folder+'Univariate2/{}_{}_uni_y{}_table.csv'.format(endpoint,dataset,k))
    elif model_type == 'LRbss':
        rest.to_csv(folder+'Univariate2/{}_{}_uni_y{}bss_lbfgs_table.csv'.format(endpoint,dataset,k))
    elif model_type == 'LRb':
        rest.to_csv(folder+'Univariate2/{}_{}_uni_y{}b_table.csv'.format(endpoint,dataset,k))
    elif model_type == 'LRss':
        rest.to_csv(folder+'Univariate2/{}_{}_uni_y{}ss_table.csv'.format(endpoint,dataset,k))
    elif model_type == 'LRnoage':
        rest.to_csv(folder+'Univariate2/{}_{}_uni_y{}noage_table.csv'.format(endpoint,dataset,k))
    elif model_type == 'cox':
        rest.to_csv(folder+'Univariate2/{}_{}_uni_cox_table.csv'.format(endpoint,dataset,k))
    elif model_type == 'AFT':
        rest.to_csv(folder+'Univariate2/{}_{}_uni_AFT_table.csv'.format(endpoint,dataset,k))
    elif model_type == 'coxstrat':
        rest.to_csv(folder+'Univariate2/{}_{}_uni_coxstrat_table.csv'.format(endpoint,dataset,k))
        
        


# In[ ]:




