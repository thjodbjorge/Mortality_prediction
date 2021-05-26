import numpy as np
from rpy2 import robjects
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
def R_pROC(resp,pred):

    rstring="""
        function(res,pred){
            library(pROC)
            out <- roc(res,pred)
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(resp),np.array(pred))
#     r_df=rfunc(resp,pred)
    return pandas2ri.OrderedDict(r_df)


def R_pROC_AUC(resp,pred):

    rstring="""
        function(res,pred){
            library(pROC)
            out <- roc(res,pred)
            ci_out <- ci(out)
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(resp),np.array(pred))
#     r_df=rfunc(resp,pred)
    return r_df

def R_pROC_compareROC(resp,pred1,pred2):

    rstring="""
        function(res,pred1,pred2){
            library(pROC)
            out <- roc.test(res,pred1,pred2)
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(resp),np.array(pred1),np.array(pred2))
#     r_df=rfunc(resp,pred)
    return pandas2ri.OrderedDict(r_df)

def R_pROC_compareROC_boot(resp,pred1,pred2):

    rstring="""
        function(res,pred1,pred2){
            library(pROC)
            out <- roc.test(res,pred1,pred2,method='bootstrap')
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(resp),np.array(pred1),np.array(pred2))
#     r_df=rfunc(resp,pred)
    return pandas2ri.OrderedDict(r_df)

def R_timeROC(tte,event,pred,time):

    rstring="""
        function(tte,event,pred,time){
            library(survival)
            library(timeROC)
            out <- timeROC(tte,event,pred,times=time,cause=1)
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(tte),np.array(event),np.array(pred),time)
#     r_df=rfunc(resp,pred)
    return pandas2ri.OrderedDict(r_df)

def R_timeROC_CI(tte,event,pred,time):

    rstring="""
        function(tte,event,pred,time){
            library(timeROC)
            out <- timeROC(tte,event,pred,times=time,cause=1,iid=TRUE)
            confint(out)
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(tte),np.array(event),np.array(pred),time)
#     r_df=rfunc(resp,pred)
    return pandas2ri.OrderedDict(r_df)

def R_timeROC_pval(tte,event,pred1,pred2,time):

    rstring="""
        function(tte,event,pred1,pred2,time){

            library(timeROC)
            out1 <- timeROC(tte,event,pred1,times=time,cause=1,iid=TRUE)
            out2 <- timeROC(tte,event,pred2,times=time,cause=1,iid=TRUE)
            compare(out1,out2)
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(tte),np.array(event),np.array(pred1),np.array(pred2),time)
#     r_df=rfunc(resp,pred)
    return pandas2ri.OrderedDict(r_df)


def R_NRIbin(resp,pred_std,pred_new, cut):

    rstring="""
        function(res,pred_std,pred_new, cut){
            library(nricens)
            out <- nribin(event=res,p.std = pred_std,p.new = pred_new, cut=cut, msg=FALSE)
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(resp),np.array(pred_std),np.array(pred_new), cut)
    return r_df

def R_NRIcens(resp,time,pred_std,pred_new, t0,cut):

    rstring="""
        function(res,time,pred_std,pred_new, t0,cut){
            library(nricens)
            out <- nricens(event=res,time=time ,p.std = pred_std,p.new = pred_new, cut=cut, msg=FALSE, t0=t0, point.method="km")
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(resp),np.array(time),np.array(pred_std),np.array(pred_new),t0, cut)
    return r_df

def R_NRIcensipw(resp,time,pred_std,pred_new, t0,cut):

    rstring="""
        function(res,time,pred_std,pred_new, t0,cut){
            library(nricens)
            out <- nricens(event=res,time=time ,p.std = pred_std,p.new = pred_new, cut=cut, msg=FALSE, t0=t0,point.method="ipw")
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(resp),np.array(time),np.array(pred_std),np.array(pred_new),t0, cut)
    return r_df

def R_censROC(resp ,time, pred,t0):

    rstring="""
        function(res,time,pred,t0){
            library(survivalROC)
            out <- survivalROC(status=res,Stime=time ,marker = pred, predict.time = t0,method='KM')
            
        }
    """
    rfunc=robjects.r(rstring)
    print(np.array(resp).shape,np.array(time).shape,np.array(pred).shape)
    r_df=rfunc(np.array(resp),np.array(time),np.array(pred),t0)
#     pandas2ri.OrderedDict(r_df)

    return r_df 


def R_hoslem(resp,pred,num=10):

    rstring="""
        function(res,pred,num){
            library(ResourceSelection)
            out <- hoslem.test(res,pred,num)
        }
    """
    rfunc=robjects.r(rstring)
    r_df=rfunc(np.array(resp),np.array(pred),num)
#     r_df=rfunc(resp,pred)
    return pandas2ri.OrderedDict(r_df)

