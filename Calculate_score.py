import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix, log_loss, brier_score_loss, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, KFold


def make_class_table(y_pred,y, bins=[0,0.075,1], disp = True):
    
#     bins = [0,0.075,1]
    pred_bins = np.digitize(y_pred,bins)
#     print(len(pred_bins))
    case_pro=[]
    pred_mean=[]
    cases = []
    controls = []
    names= []
    for i in range(1,len(bins)):
        case_pro.append((y[pred_bins==i].sum())/(y[pred_bins==i].count())) 
        pred_mean.append(y_pred[pred_bins==i].mean())
        cases.append(y[pred_bins==i].sum())
        controls.append((~y[pred_bins==i]).sum())
        names.append('{}% - {}%'.format(bins[i-1]*100,bins[i]*100))
    
    class_table = pd.DataFrame([names,pred_mean,cases,controls,case_pro]).transpose()
    class_table.columns = ['Category','Mean_pred_prob','Cases','Controls','Propor_cases']
    if disp:
        pd.options.display.float_format = '{:,.2f}'.format
        display(class_table)
    return class_table

def calculate_metrics(pred, baseline,y, bins = [0,0.075,1]):
    base_table = make_class_table(baseline,y=y,bins=bins,disp=False)
    pred_table = make_class_table(pred,y=y,bins=bins,disp = False)
    num_cases = np.sum(base_table['Cases'])
    num_controls = np.sum(base_table['Controls'])
    diff = pred_table[['Cases','Controls']]-base_table[['Cases','Controls']]
#     display(diff)
    tot = 0
    ple = 0
    for i in range(diff.shape[0]):
        ple = ple-diff.iloc[i]
#         display(diff)
#         print(ple)
        tot = tot + (ple - diff.iloc[i])

    diff = (pred-baseline)
    dcase = np.sum(diff[y])/num_cases
    dcont = np.sum(diff[~y])/num_controls
#     display(tot)
    ## AUC, Bries, logloss, IDI, NRI, AP, NRI_events, NRI_ctrl
    score = [roc_auc_score(y,pred),brier_score_loss(y,pred), log_loss(y,pred),dcase-dcont,tot[0]/num_cases -tot[1]/num_controls,average_precision_score(y,pred),tot[0]/num_cases,-tot[1]/num_controls]
    return score