import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix, log_loss, brier_score_loss, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
#from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, KFold
from boruta import BorutaPy
import lifelines as ll
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis as coxnet
from sksurv.linear_model import CoxPHSurvivalAnalysis as coxph
import sksurv 
from helpers_pat import HyperoptTuner
from glmnet import LogitNet

def predict_cv_feat(feat_cv, kf,X,y, model_type = 'lr'):
    i=0
    pred_test_cv = []
    pred_train_cv = []
    pred_feat_cv = []
    for train_index, test_index in kf.split(X):
#         print(test_index)
        feat = feat_cv[i].index
        feat = feat.drop('Age_at_sample_collection_2','sex')
        train_pred = []
        test_pred = []
        pred_feat = []
        use  = ['Age_at_sample_collection_2','sex']
        for j in range(1,len(feat)):
            use.append(feat[j])
            X_train_use = X[use]
            if model_type=='lr':
                model = LogisticRegression(penalty='none',n_jobs=-1, solver ='saga', max_iter=500, random_state=10)
            elif model_type == 'lrl2':
                Cs = np.logspace(-3,1,20)
                model = LogisticRegressionCV(penalty='l2',Cs=Cs,cv = 5,n_jobs=-1, solver ='lbfgs', scoring='neg_log_loss',random_state=10)
            else:
                print('No model')
            model.fit(X_train_use.iloc[train_index],y.iloc[train_index])
            if model_type == 'l2lr':
                print(Cs)
                print(model.C_)
            
            train_pred.append(model.predict_proba(X_train_use.iloc[train_index])[:,1])
            test_pred.append(model.predict_proba(X_train_use.iloc[test_index])[:,1])
            pred_feat.append(use)
        pred_test_cv.append(test_pred)
        pred_train_cv.append(train_pred)
        pred_feat_cv.append(pred_feat)
        i=i+1
    return [pred_feat_cv,pred_train_cv,pred_test_cv]



def predict_cv(feat, kf,X,y, model_type = 'lrl1',feat_sel_type = None):
    i=0
    pred_test_cv = []
    pred_train_cv = []
    pred_feat_cv = []
    pred_param = []
    tuned = 0
    for train_index, test_index in kf.split(X):
#         print(test_index)
        if feat_sel_type == 'boruta':    
            print('Boruta feature selection')
            np.random.seed(10)
            forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
            feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
            feat_selector.fit(X[feat].iloc[train_index].values, y.iloc[train_index].values)
            bor_feat = X[feat].loc[:,feat_selector.support_].columns.tolist()
            X_train_use = X[bor_feat]
            print(len(bor_feat))
        else:
            print('No feature selection')
            X_train_use = X[feat]
        print(X_train_use.shape)
 
        if model_type=='lr':
            model = LogisticRegression(penalty='none',n_jobs=-1, solver ='saga', max_iter=500, random_state=10)
        elif model_type == 'lrl1':
            Cs = np.logspace(-3,0,20)
            model = LogisticRegressionCV(penalty='l1',Cs=Cs,cv = 5,n_jobs=-1, solver ='saga', scoring='neg_log_loss',random_state=10)
        elif model_type == 'lrl1l2':
            Cs = np.logspace(-3,0,20)
            l1_ratios=np.linspace(0,1,10)
            model = LogisticRegressionCV(penalty='elasticnet',Cs=Cs,cv = 5,n_jobs=-1, solver ='saga', scoring='neg_log_loss',random_state=10, l1_ratios=l1_ratios)
        elif model_type =='lrl2':
            Cs = np.logspace(-3,0,20)
            model = LogisticRegressionCV(penalty='l2',Cs=Cs,cv = 5,n_jobs=-1, solver ='lbfgs', scoring='neg_log_loss',random_state=10)
            
        elif model_type == 'xgb':
            print('XGBoost')
            if not tuned:
                tuner = HyperoptTuner(xgb.XGBClassifier, 'neg_log_loss', cv=5, n_repeats=1, random_state=10)
                space = {'max_depth': [2,3],
                         'learning_rate': (0,1),
                         'n_estimators': [100,200],
                         'gamma': [0],
                         'min_child_weight': [1],
                         'max_delta_step': [0],
                         'subsample': (0.1, 1.0, 0.1),
                         'colsample_bytree': (0.1, 1.0, 0.1),
                         'reg_alpha': (0,1),
                         'reg_lambda': (0,1)}
                print('Tuning')
                parameters = tuner.optimize(X_train_use.iloc[train_index],y.iloc[train_index], space)     
                tuned = 0
                
            print('training')
            model = xgb.XGBClassifier(objective='binary:logistic' , **parameters, random_state=10) 
            
        elif model_type == 'svm':
           
            if not tuned: 
                tuner = HyperoptTuner(SVC, 'neg_log_loss', cv=5, n_repeats=1, random_state=10)
#                 space = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                  'gamma': ['scale','auto'],
#                  'C': (0,1),
#                  'degree': [2,3],
#                   'probability': [True]}
                space = {'kernel': ['linear', 'rbf'],
                 'gamma': ['scale'],
                 'C': (0,1),
                  'probability': [True]}
                print('Tuning')
                parameters = tuner.optimize(X_train_use.iloc[train_index],y.iloc[train_index], space)     
                tuned=0
            print('training')
            model = SVC(**parameters,random_state=10)
            
        elif model_type == 'mlp':
            print('XGBoost')
            if not tuned:
                tuner = HyperoptTuner(MLPClassifier, 'neg_log_loss', cv=5, n_repeats=1, random_state=10)
                space =  {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                          'alpha':(0,1),
                          'hidden_layer_sizes': [(5,),(10,),(5,5),(10,10),(5,2),(10,2)],
                          'solver': ['adam']}
                print('Tuning')
                parameters = tuner.optimize(X_train_use.iloc[train_index],y.iloc[train_index], space)     
                tuned = 0
                
            print('training')
            model = MLPClassifier(**parameters, random_state=10) 
            
            
#         elif model_type == 'mlpconst':
#             parameters =  {'hidden_layer_sizes': (2,),
#                  'activation': ['relu'],
#                  'solver': ['adam'],
#                  'alpha': (0,1)}
#             model = MLPClassifier(**parameters,random_state=10)
        else:
            print('No model')
        model.fit(X_train_use.iloc[train_index],y.iloc[train_index])
        
        if model_type == 'lrl1':

            pred_param.append(model.C_)
        if model_type == 'lrl2':

            pred_param.append(model.C_)
        if model_type == 'lrl1l2':

            pred_param.append([model.C_,model.l1_ratio_])
        if (model_type == 'xgb') | (model_type == 'mlp') | (model_type == 'svm'):
            pred_param.append([parameters])

        train_pred=[model.predict_proba(X_train_use.iloc[train_index])[:,1]]
        test_pred=[model.predict_proba(X_train_use.iloc[test_index])[:,1]]
        
        if (model_type == 'xgb') | (model_type == 'mlp'):             
            if feat_sel_type=='boruta':
                pred_feat=[pd.DataFrame(bor_feat)]
            else:
                pred_feat=[pd.DataFrame(feat)]      
        else:        
            if feat_sel_type=='boruta':
                pred_feat=[pd.DataFrame(bor_feat)[np.abs(model.coef_[0])>0]]
            else:
                pred_feat=[pd.DataFrame(feat)[np.abs(model.coef_[0])>0]]
        pred_test_cv.append(test_pred)
        pred_train_cv.append(train_pred)
        pred_feat_cv.append(pred_feat)

    
    return [pred_feat_cv,pred_train_cv,pred_test_cv,pred_param]


def predict(feat,X_train,y_train,X_test,y_test, model_type = 'lrl1'):

    if model_type=='lr':
        model = LogisticRegression(penalty='none',n_jobs=-1, solver ='saga', max_iter=500, random_state=10)
    elif model_type == 'lrl1':
        Cs = np.logspace(-2,0,20)
        model = LogisticRegressionCV(penalty='l1',Cs=Cs,cv = 5,n_jobs=-1, solver ='saga', scoring='neg_log_loss',random_state=10) # ATH liblinear penalizes the constant
    elif model_type == 'lrl1l2':
        Cs = np.logspace(-3,0,20)
        l1_ratios=np.linspace(0,1,10)
        model = LogisticRegressionCV(penalty='elasticnet',Cs=Cs,cv = 5,n_jobs=-1, solver ='saga', scoring='neg_log_loss',random_state=10, l1_ratios=l1_ratios)
    elif model_type =='lrl2':
        Cs = np.logspace(-3,1,20)
        model = LogisticRegressionCV(penalty='l2',Cs=Cs,cv = 5,n_jobs=-1, solver ='lbfgs', scoring='neg_log_loss',random_state=10)
        
    elif model_type =='glmnetl1':
        Cs = np.logspace(-3,1,20)
        model = LogitNet(n_splits=5,n_jobs=-1,random_state=10,scoring='roc_auc')
        
    else:
        print('No model')
    model.fit(X_train[feat],y_train)
        
    if model_type == 'lrl1':
        print(Cs)
        print(model.C_)
    if model_type == 'lrl1l2':
        print(Cs)
        print(l1_ratios)
        print(model.C_)
        print(model.l1_ratio_)
        
    train_pred= model.predict_proba(X_train[feat])[:,1]
    test_pred= model.predict_proba(X_test[feat])[:,1]
    model = model
        
    return model,train_pred,test_pred, feat



def predict_new(feat,X_train,y_train,X_test,y_test, model_type = 'lrl1'):
    ### The only difference from predict is the number of iterations in the penalized models.
    if model_type=='lr':
        model = LogisticRegression(penalty='none',n_jobs=-1, solver ='saga', max_iter=500, random_state=10)
    elif model_type == 'lrl1':
        Cs = np.logspace(-2,0,20)
        model = LogisticRegressionCV(penalty='l1',Cs=Cs,cv = 5,n_jobs=-1, solver ='saga', scoring='neg_log_loss',random_state=10,max_iter=500) # ATH liblinear penalizes the constant
    elif model_type == 'lrl1l2':
        Cs = np.logspace(-3,0,20)
        l1_ratios=np.linspace(0,1,10)
        model = LogisticRegressionCV(penalty='elasticnet',Cs=Cs,cv = 5,n_jobs=-1, solver ='saga', scoring='neg_log_loss',random_state=10, l1_ratios=l1_ratios,max_iter=500)
    elif model_type =='lrl2':
        Cs = np.logspace(-3,1,20)
        model = LogisticRegressionCV(penalty='l2',Cs=Cs,cv = 5,n_jobs=-1, solver ='lbfgs', scoring='neg_log_loss',random_state=10, max_iter=500)
        
    elif model_type =='glmnetl1':
        Cs = np.logspace(-3,1,20)
        model = LogitNet(n_splits=5,n_jobs=-1,random_state=10,scoring='roc_auc')
        
    else:
        print('No model')
    model.fit(X_train[feat],y_train)
        
    if model_type == 'lrl1':
        print(Cs)
        print(model.C_)
    if model_type == 'lrl1l2':
        print(Cs)
        print(l1_ratios)
        print(model.C_)
        print(model.l1_ratio_)
        
        
    train_pred= model.predict_proba(X_train[feat])[:,1]
    test_pred= model.predict_proba(X_test[feat])[:,1]
    model = model
        
    
    return model,train_pred,test_pred, feat





def predict_surv_cv2(feat,kf,X,y_surv, y = '', k=9, event_col = 'event', time_to_event_col = 'time_to_event', model_type = 'coxl2',feat_sel_type = None, alpha = 0.001,penalty_factor=None):
    i=0
    pred_test_cv = []
    pred_train_cv = []
    pred_feat_cv = []
    pred_param = []
    pred_models = []
    for train_index, test_index in kf.split(X):
#         print(test_index)
        if feat_sel_type == 'boruta':    
            print('Boruta feature selection')
            np.random.seed(10)
            forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
            feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
            feat_selector.fit(X[feat].iloc[train_index].values, y.iloc[train_index].values)
            bor_feat = X[feat].loc[:,feat_selector.support_].columns.tolist()
            X_train_use = X[bor_feat]
            print(len(bor_feat))
        else:
            print('No feature selection')
            X_train_use = X[feat]

        if model_type == 'coxl2':
#             y_surv = Surv.from_dataframe('event','time_to_event',ysurv_train)
            premodel = coxnet( fit_baseline_model=True, l1_ratio = 0.001, alpha_min_ratio=0.0001).fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
            GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
            GSmodel.fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
            model = GSmodel.best_estimator_
            
            
            results = pd.DataFrame(GSmodel.cv_results_)
            alphas = results.param_alphas.map(lambda x: x[0])

            
        elif model_type == 'coxl1':
#             y_surv = Surv.from_dataframe('event','time_to_event',ysurv_train)
            premodel = coxnet( fit_baseline_model=True, l1_ratio = 1).fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
            GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
            GSmodel.fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
            model = GSmodel.best_estimator_
            
            results = pd.DataFrame(GSmodel.cv_results_)
            alphas = results.param_alphas.map(lambda x: x[0])

        elif model_type == 'coxl1l2':
            premodel = coxnet( fit_baseline_model=True, l1_ratio = 0.5).fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
            GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
            GSmodel.fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
            model = GSmodel.best_estimator_
            
            results = pd.DataFrame(GSmodel.cv_results_)
            alphas = results.param_alphas.map(lambda x: x[0])
        elif model_type == 'coxl1l2cv':
            ratios = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1]
#             ratios = [0.001,0.5,1]
            GSmodels = []
            GS_score = []
            for l1_ratio in ratios:
                print(l1_ratio)
                premodel = coxnet( fit_baseline_model=True, l1_ratio = l1_ratio).fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
                GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
                GSmodel.fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
                GS_score.append(GSmodel.best_score_)
                GSmodels.append(GSmodel)
                print(GSmodel.best_score_)
            j = np.argmax(GS_score)
            GSmodel = GSmodels[j]
            model = GSmodel.best_estimator_
            
            results = pd.DataFrame(GSmodel.cv_results_)
            alphas = results.param_alphas.map(lambda x: x[0])
            
        elif model_type == 'cox':
            model = coxph()
            model.fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
            GSmodel = model
            
        elif model_type == 'coxl1const':
#             y_surv = Surv.from_dataframe('event','time_to_event',ysurv_train)
            model = coxnet( fit_baseline_model=True, l1_ratio = 1, alphas = [alpha],penalty_factor = penalty_factor).fit(X_train_use.iloc[train_index],
                                                                                         Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
            GSmodel = model
#             GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
#             GSmodel.fit(X_train_use.iloc[train_index],Surv.from_dataframe(event_col,time_to_event_col,y_surv.iloc[train_index]))
#             model = GSmodel.best_estimator_
            
            results = pd.DataFrame(GSmodel.cv_results_)
            alphas = results.param_alphas.map(lambda x: x[0])
            
          
            
        else:
            print('No model')
        
        print('Model fit')
        if model_type == 'cox':
            pred_param.append([0,0])
        else:
            pred_param.append([alphas ,GSmodel.best_params_])
                
        train_pred = [1 - p(k+1) for p in model.predict_survival_function(X_train_use.iloc[train_index])]
        test_pred  = [1 - p(k+1) for p in model.predict_survival_function(X_train_use.iloc[test_index])]
    
        
        if feat_sel_type=='boruta':
            pred_feat=[pd.DataFrame(bor_feat)[np.abs(model.coef_)>0]]
        else:
            pred_feat=[pd.DataFrame(feat)[np.abs(model.coef_)>0]]
        
        pred_test_cv.append([test_pred])
        pred_train_cv.append([train_pred])
        pred_feat_cv.append(pred_feat)
        pred_models.append(GSmodel)

    
    return [pred_feat_cv,pred_train_cv,pred_test_cv,pred_param, pred_models]

def predict_surv_cv(feat,kf,X,tte,y,k,event_col='event', model_type = 'coxl2',feat_sel_type = None):
    CoxRegression = sklearn_adapter(ll.CoxPHFitter, event_col = event_col)
    AFTmodel = sklearn_adapter(ll.LogLogisticAFTFitter, event_col = event_col)
    i=0
    pred_test_cv = []
    pred_train_cv = []
    pred_feat_cv = []
    pred_param = []
    pred_models = []
    X_train_use = X[feat]
    for train_index, test_index in kf.split(X):
#         print(test_index)
        if model_type == 'coxl2':
            model = CoxRegression()
            GSmodel = GridSearchCV(model,param_grid = {'penalizer': np.logspace(-1,3,20),'l1_ratio':[0]},cv=5)
            GSmodel.fit(X_train_use.iloc[train_index],tte.iloc[train_index])
            model = GSmodel.best_estimator_
            
        elif model_type == 'coxl1':
            model = CoxRegression()
            GSmodel = GridSearchCV(model,param_grid = {'penalizer': np.logspace(-1,1,10),'l1_ratio': [1]},cv=5)
            GSmodel.fit(X_train_use.iloc[train_index],tte.iloc[train_index])
            model = GSmodel.best_estimator_
            
        elif model_type == 'coxl1l2':
            model = CoxRegression()
            GSmodel = GridSearchCV(model,param_grid = {'penalizer': np.logspace(-1,2,10),'l1_ratio': [0,0.1,0.5,0.9]},cv=5)
            GSmodel.fit(X_train_use.iloc[train_index],tte.iloc[train_index])
            model = GSmodel.best_estimator_
            
        elif model_type == 'AFTl2':         
            
            model = AFTmodel(fit_intercept = True)
            GSmodel = GridSearchCV(model,param_grid = {'penalizer': np.logspace(-2,0,20),'l1_ratio':[0]},cv=5,verbose = 1)
            GSmodel.fit(X_train_use.iloc[train_index],tte.iloc[train_index])
            model = GSmodel.best_estimator_
            
        elif model_type == 'AFTl1':         
            
            model = AFTmodel(fit_intercept = True)
            GSmodel = GridSearchCV(model,param_grid = {'penalizer': np.logspace(-3,-1,20),'l1_ratio':[1]},cv=5,verbose = 1)
            GSmodel.fit(X_train_use.iloc[train_index],tte.iloc[train_index])
            model = GSmodel.best_estimator_
            
        elif model_type == 'AFT':         
            
            model = AFTmodel(fit_intercept = True)
            GSmodel = GridSearchCV(model,param_grid = {'penalizer': [0],'l1_ratio':[0]},cv=5,verbose = 1)
            GSmodel.fit(X_train_use.iloc[train_index],tte.iloc[train_index])
            model = GSmodel.best_estimator_  

        else:
            print('No model')
        
        
        
        pred_param.append(GSmodel.best_params_)
 
        train_pred=[1 - model.lifelines_model.predict_survival_function(X_train_use.iloc[train_index],times = [k+1])]
        test_pred=[1 - model.lifelines_model.predict_survival_function(X_train_use.iloc[test_index],times = [k+1])]
#         pred_feat=[X_train_use.columns]
        

        pred_feat = pd.DataFrame(feat[1:])[list(np.abs(model.lifelines_model.params_[:-2])>0)]
                
        pred_test_cv.append(test_pred)
        pred_train_cv.append(train_pred)
        pred_feat_cv.append(pred_feat)
        pred_models.append('Pickling not yet supported')
    
    return [pred_feat_cv,pred_train_cv,pred_test_cv,pred_param,pred_models]



def predict_surv2(feat,X,y_surv, y = '', k=9 ,event_col='event', time_to_event_col = 'time_to_event', model_type = 'coxl2',l1_ratio = 0.5,feat_sel_type = None, alpha=0.001,penalty_factor=None):

    i=0

    if feat_sel_type == 'boruta':    
        print('Boruta feature selection')
        np.random.seed(10)
        forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
        feat_selector = BorutaPy(forest, n_estimators='auto', verbose=0, max_iter=200)
        feat_selector.fit(X[feat].values, y.values)
        bor_feat = X[feat].loc[:,feat_selector.support_].columns.tolist()
        X_train_use = X[bor_feat]
        print(len(bor_feat))
    else:
        print('No feature selection')
        X_train_use = X[feat]
    ## Grid searchCV uses the estimators score method by default which is the concordance index in this case.
    if model_type == 'coxl2':
        premodel = coxnet( fit_baseline_model=True, l1_ratio = 0.001, alpha_min_ratio=0.0001).fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
        GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
        GSmodel.fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
        model = GSmodel.best_estimator_
                 
    elif model_type == 'coxl1':
        premodel = coxnet( fit_baseline_model=True, l1_ratio = 1, alpha_min_ratio=0.0001,penalty_factor=penalty_factor).fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
        GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
        GSmodel.fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
        model = GSmodel.best_estimator_
        
    elif model_type == 'coxl1l2':
        premodel = coxnet( fit_baseline_model=True, l1_ratio = l1_ratio, alpha_min_ratio=0.0001).fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
        GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
        GSmodel.fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
        model = GSmodel.best_estimator_
    elif model_type == 'coxl1l2cv':
        ratios = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1]
        GSmodels = []
        GS_score = []
        for l1_ratio in ratios:
            print(l1_ratio)
            premodel = coxnet( fit_baseline_model=True, l1_ratio = l1_ratio).fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
            GSmodel = GridSearchCV(premodel,param_grid = {'alphas': [[v] for v in premodel.alphas_]},cv=5, n_jobs=-1)
            GSmodel.fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
            GS_score.append(GSmodel.best_score_)
            GSmodels.append(GSmodel)
            print(GSmodel.best_score_)
        j = np.argmax(GS_score)
        GSmodel = GSmodels[j]
        model = GSmodel.best_estimator_         
        
        
    elif model_type == 'cox':
        model = coxph()
        model.fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
        GSmodel = model
        
    elif model_type == 'coxl1const':
        model = coxnet( fit_baseline_model=True, l1_ratio = 1, alphas = [alpha],penalty_factor=penalty_factor).fit(X_train_use,Surv.from_dataframe(event_col,time_to_event_col,y_surv))
        GSmodel = model
        
    else:
        print('No model')
 

    train_pred = [1 - p(k+1) for p in model.predict_survival_function(X_train_use)]
    
    pred_train = train_pred
    pred_feat = X_train_use.columns
    pred_model = model
    models = GSmodel
    
    return pred_model,pred_train,models,pred_feat


def predictions_surv_cv(models,feat,kf,X,k):
    i=0
    pred_test_cv = []
    pred_train_cv = []
    pred_feat_cv = []
    pred_models = []

    for train_index, test_index in kf.split(X):
        X_use = X[feat]
        model = models[i]
#         feat = features[i]
        print(model)
        i = i+1
    
        train_pred = [1 - p(k+1) for p in model.predict_survival_function(X_use.iloc[train_index],alpha=model.alphas_[0])]
        test_pred  = [1 - p(k+1) for p in model.predict_survival_function(X_use.iloc[test_index],alpha = model.alphas_[0])]
        pred_feat=[pd.DataFrame(feat)[np.abs(model.coef_)>0]]  
        
        pred_test_cv.append([test_pred])
        pred_train_cv.append([train_pred])
        pred_feat_cv.append(pred_feat)
        pred_models.append(model)
    
    return [pred_feat_cv,pred_train_cv,pred_test_cv,'Nothing', pred_models]


def predictions_surv(model,feat,X,k):
    X_use = X[feat]
    pred_risk = [1 - p(k+1) for p in model.predict_survival_function(X_use)]
    pred_linear = model.predict(X_use)
    return [pred_risk, pred_linear]


