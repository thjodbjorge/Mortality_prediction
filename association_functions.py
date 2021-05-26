import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t,ttest_ind,sem, pearsonr,spearmanr
import seaborn as sns
import matplotlib.pyplot as plt


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def group_assoc(X, pred, split, corr_col = ['age','sex','age2','agesex','age2sex'],split_name = 'statin' ,show_plot=False):
    print('Correct for age and sex')
    pred2 = pred-sm.OLS(pred,sm.add_constant(X[corr_col])).fit().predict(sm.add_constant(X[corr_col]))
    
    print('Number of {} users: '.format(split_name), pred2[split].shape)
    print('Not using {} : '.format(split_name), pred2[~split].shape)
    print('Risk for {} users: '.format(split_name), pred2[split].mean())
    print('Risk for not using {}: '.format(split_name), pred2[~split].mean())
    print('T-test: ', ttest_ind(pred2[split],pred2[~split]))
    
    print('Correct for nothing')

    print('{} users: '.format(split_name), pred[split].mean())
    print('Not using {}: '.format(split_name), pred[~split].mean())
    
    print('Match age and sex')
    group_split_all = []
    group_nosplit_all = []

    age=X['age']
    gsm = age[split&~X['sex']]
    gnosm =  age[~split&~X['sex']]

    gsm_cat = list(np.digitize(gsm,bins = np.arange(20,100,1)))
    gnosm_cat = np.digitize(gnosm,bins = np.arange(20,100,1))

    not_in_list = []
    in_list = []
    other_list = []
    for j, cat in enumerate(gnosm_cat):
        try:
            i = gsm_cat.index(cat)
            in_list.append(j)
            other_list.append(i)
            gsm_cat[i] = np.nan
        except:
            not_in_list.append(cat)

    print('Men')
    print(gsm.iloc[other_list].mean())
    print(gnosm.iloc[in_list].mean())
    if show_plot:
        gsm.iloc[other_list].hist()
        plt.show()
        gnosm.iloc[in_list].hist()
        plt.show()

    gs = pred[split&~X['sex']]
    gnos =  pred[~split&~X['sex']]
    
    group_split_all.extend(gs[other_list])
    group_nosplit_all.extend(gnos[in_list])
    print(gs[other_list].mean())
    print(gnos[in_list].mean())
    
    gsw = age[split&X['sex']]
    gnosw =  age[~split&X['sex']]

    gsw_cat = list(np.digitize(gsw,bins = np.arange(20,100,1)))

    not_in_list = []
    in_list = []
    other_list = []
    for j, cat in enumerate(gnosw_cat):
        try:
            i = gsw_cat.index(cat)
            in_list.append(j)
            other_list.append(i)
            gsw_cat[i] = np.nan
        except:
            not_in_list.append(cat)

    print('Women')
    print(gsw.iloc[other_list].mean())
    print(gnosw.iloc[in_list].mean())
    if show_plot:
        gsw.iloc[other_list].hist()
        plt.show()
        gnosw.iloc[in_list].hist()
        plt.show()

    gs = pred[split&X['sex']]
    gnos =  pred[~split&X['sex']]
    
    group_split_all.extend(gs[other_list])
    group_nosplit_all.extend(gnos[in_list])
    print(gs[other_list].mean())
    print(gnos[in_list].mean())
    
    print('Use {}: '.format(split_name),len(group_split_all), np.mean(group_split_all))
    print('No {}: '.format(split_name), len(group_nosplit_all), np.mean(group_nosplit_all))
    print(ttest_ind(group_split_all,group_nosplit_all))
    
    fig = plt.figure(figsize=[12,6])
    ax1 = fig.add_subplot(1,2,1)
    sns.pointplot(data=[group_split_all,group_nosplit_all],join=False,ax=ax1)
    ax1.set_xticklabels(['{}'.format(split_name),'No {}'.format(split_name)])
    ax1.set_ylabel('Predicted risk')
    ax1.set_title('Age and sex matched groups')
    ax2 = fig.add_subplot(1,2,2)
    sns.pointplot(data=[pred2[split],pred2[~split]],join=False,ax=ax2)
    ax2.set_xticklabels(['{}'.format(split_name),'No {}'.format(split_name)])
    ax2.set_ylabel('Corrected predicted risk')
    ax2.set_title('Corrected for covariates')
    plt.show()
    
    
    return group_split_all, group_nosplit_all,pred2[split],pred2[~split]#, [gsm.iloc[other_list], gnosm.iloc[in_list],gsw.iloc[other_list], gnosw.iloc[in_list]]
#     print(t().interval(0.95,group_split_all))
#     print(t.interval(0.95,group_nosplit_all))
    
    
def associations(X,col_names, covariates, I_use, pred_type, categorical=False, check_age=False):
    association = []
    for col1 in col_names:
        ind = X.loc[I_use,col1].dropna().index
#         print(col1)
        if len(ind) < 1000:
            continue
        if categorical:
            split = X.loc[ind,col1].astype(bool)
            if np.sum(split)<10:
                continue
            ass_res = [col1,np.sum(~split),np.sum(split)]
            ass_res_names = ['Phenotype', 'Control','Case']         
        else:
            ass_res = [col1,len(ind)]
            ass_res_names = ['Phenotype', 'Number']
        try:
            columns = [col1]
            columns.extend(covariates)

            model = sm.OLS(X.loc[ind,pred_type['org']],sm.add_constant(X.loc[ind,columns],prepend=False))
            res = model.fit(disp = 0)
            ass_res.extend([res.params[0],res.pvalues[0]])
            ass_res_names.extend(['beta_phen','pval_phen'])


            columns = [pred_type['org']]
            columns.extend(covariates)
            if categorical:
                model = sm.Logit(X.loc[ind,col1],sm.add_constant(X.loc[ind,columns],prepend=False))
                res = model.fit(disp = 0)
                ass_res.extend([res.params[0],res.pvalues[0]])
                ass_res_names.extend(['logit_beta_pred','logit_pval_pred'])

            else:
                model = sm.OLS(X.loc[ind,col1],sm.add_constant(X.loc[ind,columns],prepend=False))
                res = model.fit(disp = 0)
                ass_res.extend([res.params[0],res.pvalues[0]])
                ass_res_names.extend(['beta_pred','pval_pred'])

            ## Standardized variables
            columns = [col1]
            columns.extend(covariates)

            model = sm.OLS(X.loc[ind,pred_type['std']],sm.add_constant(X.loc[ind,columns],prepend=False))
            res = model.fit(disp = 0)
            ass_res.extend([res.params[0],res.pvalues[0]])
            ass_res_names.extend(['beta_phen_std','pval_phen_std'])

            columns = [pred_type['std']]
            columns.extend(covariates)

            if categorical:
                model = sm.Logit(X.loc[ind,col1],sm.add_constant(X.loc[ind,columns],prepend=False))
                res = model.fit(disp = 0)
                ass_res.extend([res.params[0],res.pvalues[0]])
                ass_res_names.extend(['logit_beta_pred_std','logit_pval_pred_std'])

            else:
                model = sm.OLS(X.loc[ind,col1],sm.add_constant(X.loc[ind,columns],prepend=False))
                res = model.fit(disp = 0)
                ass_res.extend([res.params[0],res.pvalues[0]])
                ass_res_names.extend(['beta_pred_std','pval_pred_std'])
        except:
            print(col1)
            continue
        ## Correlation for corrected version
        
        pcorr= pearsonr(X.loc[ind,pred_type['corr']],X.loc[ind,col1])
        ass_res.extend([pcorr[0],pcorr[1]])
        ass_res_names.extend(['pearson_coef','pearson_pval'])
        
        pcorr= pearsonr(X.loc[ind,pred_type['org']],X.loc[ind,col1])
        ass_res.extend([pcorr[0],pcorr[1]])
        ass_res_names.extend(['org_pearson_coef','org_pearson_pval'])
        
        pcorr= pearsonr(X.loc[ind,'age'],X.loc[ind,col1])
        ass_res.extend([pcorr[0],pcorr[1]])
        ass_res_names.extend(['age_pearson_coef','age_pearson_pval'])

        scorr = spearmanr(X.loc[ind,pred_type['corr']],X.loc[ind,col1])
        ass_res.extend([scorr[0],scorr[1]])
        ass_res_names.extend(['Spearman_coef','Spearman_pval'])
        
        scorr = spearmanr(X.loc[ind,'age'],X.loc[ind,col1])
        ass_res.extend([scorr[0],scorr[1]])
        ass_res_names.extend(['age_Spearman_coef','age_Spearman_pval'])
        
        if categorical:

            phen_true = X.loc[ind,pred_type['corr']][split]
            phen_false = X.loc[ind,pred_type['corr']][~split]
#             print('T-test: ', ttest_ind(phen_true,phen_false))
            ass_res.extend([phen_true.mean()-phen_false.mean(), ttest_ind(phen_true,phen_false)[1]])
            ass_res_names.extend(['diff_mean','ttest_pval'])
        
            phen_true = X.loc[ind,'age'][split]
            phen_false = X.loc[ind,'age'][~split]
#             print('T-test: ', ttest_ind(phen_true,phen_false))
            ass_res.extend([phen_true.mean()-phen_false.mean(), ttest_ind(phen_true,phen_false)[1]])
            ass_res_names.extend(['age_diff_mean','age_ttest_pval'])
        
        if check_age:
            pcorr= pearsonr(X.loc[ind,pred_type['corr']],X.loc[ind,'age'])
            ass_res.extend([pcorr[0],pcorr[1]])
            ass_res_names.extend(['age_pred_pearson_coef','age_pred_pearson_pval'])
            
            pcorr= pearsonr(X.loc[ind,pred_type['org']],X.loc[ind,'age'])
            ass_res.extend([pcorr[0],pcorr[1]])
            ass_res_names.extend(['age_pred_org_pearson_coef','age_pred_org_pearson_pval'])
            
            pcorr= pearsonr(X.loc[ind,pred_type['corr']],X.loc[ind,'age2'])
            ass_res.extend([pcorr[0],pcorr[1]])
            ass_res_names.extend(['age2_pred_pearson_coef','age2_pred_pearson_pval'])
            
            pcorr= pearsonr(X.loc[ind,pred_type['corr']],X.loc[ind,'age3'])
            ass_res.extend([pcorr[0],pcorr[1]])
            ass_res_names.extend(['age3_pred_pearson_coef','age3_pred_pearson_pval'])
            
            pcorr= pearsonr(X.loc[ind,pred_type['corr']],X.loc[ind,'lnage'])
            ass_res.extend([pcorr[0],pcorr[1]])
            ass_res_names.extend(['lnage_pred_pearson_coef','lnage_pred_pearson_pval'])


            scorr = spearmanr(X.loc[ind,pred_type['corr']],X.loc[ind,'age'])
            ass_res.extend([scorr[0],scorr[1]])
            ass_res_names.extend(['age_pred_Spearman_coef','age_pred_Spearman_pval'])
        

        association.append(ass_res)
    ass_df = pd.DataFrame(association,columns=ass_res_names)
    ass_df.set_index('Phenotype',inplace=True) 
    return ass_df