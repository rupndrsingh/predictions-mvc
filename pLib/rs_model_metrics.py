# Custom module to assess performance of ML model
# Author: Rupinder Singh (Oct. 25, 2016)

from __future__ import division
import matplotlib.pyplot as plt
from sklearn.metrics import auc,r2_score
import numpy as np
import pandas as pd
from scipy.stats import sem
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import classification_report
import os

def get_model_coef(model,columns,export_fn='./Data/model_coef.csv'):
    # Returns coefficient of scikit learn model
    # model: scikit model object
    # columns: array of strings associated with each coefficient
    # export_fn: filename to export csv of coefficients
    i = pd.Series(model.intercept_, index=['intercept'])
    c = pd.Series(model.coef_[0,:], index=columns)
    model_coef = pd.concat([i,c]).sort_values()
    model_coef = model_coef[model_coef != 0] #remove zero coef
    model_coef.to_csv(export_fn)
    return model_coef

def compute_classification_score(model,y_train,X_train,y_test,X_test,event):
    y_pred = model.predict(X_test)
    target_names = ['no_'+event, event]
    print(classification_report(y_test, y_pred, target_names=target_names))

    acc_train = model.score(X_train,y_train) 
    acc_test = model.score(X_test,y_test) 

    tbl=[['','Train','Test'],
    ['Incidence','{0:.2f}%'.format(y_train.mean()*100),'{0:.2f}%'.format(y_test.mean()*100)],
    ['Actual ACC','{0:.2f}'.format(acc_train),'{0:.2f}'.format(acc_test)]]
    return tbl

def compute_rsq(y_pred,y_true):
    """
    Computes coefficient of determination
    Inputs are two numpy arrays
    """
    ss_res = sum((y_true - y_pred)**2)
    ss_tot = sum((y_true - y_true.mean())**2)
    rsq = 1-ss_res/ss_tot
    rsq = r2_score(y_true,y_pred)
    return rsq

def compute_metrics(y_pred,y_true):
    # Function computes metrics using predicted and true labels
    ap = (y_true==1).sum() #actual positives
    an = (y_true==0).sum() #actual negatives
    pp = (y_pred==1).sum() #predicted positives
    pn = (y_pred==0).sum() #predicted negatives 
    tn = ((y_pred==0) & (y_true==0)).sum()
    fp = ((y_pred==1) & (y_true==0)).sum()
    fn = ((y_pred==0) & (y_true==1)).sum()
    tp = ((y_pred==1) & (y_true==1)).sum()
    tnr = tn / an
    fpr = fp / an
    fnr = fn / ap
    tpr = tp / ap
    pr = tp / pp 
    ptp = tp/(ap+an)
    pfp = fp/(ap+an)
    ptn = tn/(ap+an)
    pfn = fn/(ap+an)
    return tnr,fpr,fnr,tpr,pr,ptp,pfp,ptn,pfn

def model_metrics(model,y_true,X_true,Nsets):
    # Function produces metric arrarys that have nth rows and nsplits columns,
    # where nth is the number of thresholds used and Nsets is number of test sets created
    n=int(y_true.shape[0]/Nsets)
    indices = np.random.permutation(y_true.shape[0])
    idx = ()
    for i in range(0,Nsets):
        idx += (indices[i*n:(i+1)*n],)

    thL = np.arange(0,1.01,0.01) # list of thresholds
    Nrows = len(thL)

    tnrL=np.zeros((Nrows,Nsets))
    fprL=np.zeros((Nrows,Nsets))
    fnrL=np.zeros((Nrows,Nsets))
    tprL=np.zeros((Nrows,Nsets))
    PrL=np.zeros((Nrows,Nsets))
    Ptp=np.zeros((Nrows,Nsets))
    Pfp=np.zeros((Nrows,Nsets))
    Ptn=np.zeros((Nrows,Nsets))
    Pfn=np.zeros((Nrows,Nsets))
    n = 0
    for idx_n in idx:
        yprob = model.predict_proba(X_true.values[idx_n,:]) #issue 
        m = 0
        for th in thL:
            y_pred=yprob[:,1]>th
            (tnrL[m,n],fprL[m,n],fnrL[m,n],tprL[m,n],PrL[m,n],\
             Ptp[m,n],Pfp[m,n],Ptn[m,n],Pfn[m,n])=compute_metrics(y_pred,y_true.values[idx_n])
            m+=1
        n+=1

    PrL[np.isnan(PrL)] = 0 # only needed for this as pp could be 0
    return thL,tnrL,fprL,fnrL,tprL,PrL,Ptp,Pfp,Ptn,Pfn

def update_plot_parameters(fnt_sz=14,lbl_sz=16):
    plot_parameters = {'font.size': fnt_sz,'text.usetex':True,'axes.titlesize':lbl_sz,
                        'legend.frameon':True,'legend.framealpha':1,'legend.fontsize':fnt_sz,
                        'axes.labelsize':lbl_sz,'axes.facecolor':'white','axes.edgecolor':'lightgray',
                        'axes.linewidth':1.0,'axes.grid':True,
                        'grid.color':'lightgray','figure.edgecolor':(1,1,1,1),
                        'xtick.labelsize':fnt_sz,'ytick.labelsize':fnt_sz}
    plt.rcParams.update(plot_parameters)

def compute_savings(Ptp,Pfp,parameters={'eta':0.55,'Ci':104,'Ce':14393,'Cne':5840},confidence=1.95):
    # Computes expected cost savings using probability of true-positive (Ptp),
    # probability of false-positive (Pfp), probability of successful intervention (eta),
    # cost of intervention (Ci), cost of event (Ce), cost of no event (Cne)
    Nrows,Nsets = Ptp.shape
    eta = parameters['eta']; Ci = parameters['Ci']; Ce = parameters['Ce']; Cne = parameters['Cne']

    csL = np.zeros((Nrows,Nsets))
    for i in range(0,Nsets):
        csL[:,i]=Ptp[:,i]*eta*(Ce-Cne)-(Ptp[:,i]+Pfp[:,i])*Ci
    csM=csL.mean(axis=1)
    csE=sem(csL,axis=1)*confidence
    return csL,csM,csE

def plot_model_metrics(thL,fprL,tprL,PrL,export_fn='./Figures/model_metrics.eps',confidence=1.95):

    Nrows,Nsets = tprL.shape #number of columns equal number of test sets
    th = 0.5
    i=np.where(thL==th)
    fm=fprL.mean(axis=1)[i]
    tm=tprL.mean(axis=1)[i]
    pm=PrL.mean(axis=1)[i]
    fe=sem(fprL,axis=1)[i]*confidence
    te=sem(tprL,axis=1)[i]*confidence
    pe=sem(PrL,axis=1)[i]*confidence

    fnt_sz=14; lbl_sz=16;
    update_plot_parameters(fnt_sz=fnt_sz,lbl_sz=lbl_sz)
    fig,ax = plt.subplots(1,1,sharex=True,sharey=False,figsize=[15,8])
    plt.tight_layout(pad=2, w_pad=5, h_pad=0)

    for n in range(0,Nsets):
        ax.plot(thL,tprL[:,n],label=None,
                linestyle='--',color='grey')
        ax.plot(thL,fprL[:,n],label=None,
                linestyle='--',color='grey')
        ax.plot(thL,PrL[:,n],label=None,
                linestyle='--',color='grey')
    ax.plot(thL,tprL.mean(axis=1),label=r'recall',
            linestyle='-',color='black')
    ax.plot(thL,fprL.mean(axis=1),label=r'fpr',
            linestyle='-',color='black')
    ax.plot(thL,PrL.mean(axis=1),label=r'precision',
            linestyle='-',color='black')

    ax.set_ylabel('rate')
    ax.set_xlabel('threshold')

    #Annotation Shifts
    scalex = 0.03*(np.max(thL)-np.min(thL))
    scaley = 0.03*(1-0)

    ax.annotate('*',(th,tm),backgroundcolor='white',fontsize=fnt_sz+10,alpha=0.8) 
    ax.annotate('*',(th,fm),backgroundcolor='white',fontsize=fnt_sz+10,alpha=0.8) 
    ax.annotate('*',(th,pm),backgroundcolor='white',fontsize=fnt_sz+10,alpha=0.8) 
    ax.annotate('recall ($%.2f \pm %.2f$)'%(tm,te),(th-scalex,tm-scaley),
                   fontsize=fnt_sz,backgroundcolor='white',alpha=0.8) 
    ax.annotate('fpr ($%.2f \pm %.2f$)'%(fm,fe),(th-scalex,fm-scaley),
                   fontsize=fnt_sz,backgroundcolor='white',alpha=0.8) 
    ax.annotate('precision ($%.2f \pm %.2f$)'%(pm,pe),(th-scalex,pm-scaley),
                   fontsize=fnt_sz,backgroundcolor='white',alpha=0.8) 
    
    directory = os.path.dirname(export_fn) #create directory if it doesn't exist
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(export_fn, format='eps', dpi=1000,facecolor=fig.get_facecolor(), edgecolor='none')


def plot_roc_roi(thL,fprL,tprL,Ptp,Pfp,export_fn='./Figures/roc_savings.eps',parameters={'eta':0.55,'Ci':104,'Ce':14393,'Cne':5840},confidence=1.95):

    Nrows,Nsets = tprL.shape #number of columns equal number of test sets

    fnt_sz=14; lbl_sz=16;
    update_plot_parameters(fnt_sz=fnt_sz,lbl_sz=lbl_sz)
    fig,ax = plt.subplots(1,2,sharex=True,sharey=False,figsize=[15,8])
    plt.tight_layout(pad=2, w_pad=5, h_pad=0)

    # ROI
    csL,csM,csE = compute_savings(Ptp,Pfp,parameters=parameters)
    j=np.argmax(csM)
    th = thL[j] #threshold for annotation
    for i in range(0,Nsets):
        ax[1].plot(thL,csL[:,i],label='TS%i ($%.2f$, $\$%.0f$)'%(i+1,thL[j],csL[j,i]),
               linestyle='--',color='grey')
    ax[1].plot(thL,csM,label='TSM ($%.2f$, $\$%.0f\pm%.0f$)'%(thL[j],csM[j],csE[j]),
               linestyle='-',color='black')
    ax[1].axvline(th,linestyle=':',color='gray') 

    ax[1].legend(loc='lower right')
    ax[1].set_ylabel(r'$E(c_{saving})$')
    ax[1].set_xlabel(r'Threshold ($E(c_i) = \$%.f$, $\eta = %.2f$)'%(parameters['Ci'],parameters['eta']))

    # ROC
    aucL = np.zeros((Nsets,1))
    for i in range(0,Nsets):
        aucL[i]=auc(fprL[:,i], tprL[:,i])    
    aucM=auc(fprL.mean(axis=1), tprL.mean(axis=1))
    aucE=sem(aucL)*confidence

    fm=fprL.mean(axis=1)[j]
    tm=tprL.mean(axis=1)[j]
    fe=sem(fprL,axis=1)[j]*confidence
    te=sem(tprL,axis=1)[j]*confidence

    for i in range(0,Nsets):
        ax[0].plot(fprL[:,i],tprL[:,i],label='TS%i ($%.2f$)'%(i+1,aucL[i]),
                linestyle='--',color='grey')
    ax[0].plot(fprL.mean(axis=1),tprL.mean(axis=1),label=r'TSM ($%.2f\pm%.2f$)'%(aucM,aucE),
            linestyle='-',color='black')
    ax[0].legend(loc='lower right')
    ax[0].set_ylabel('recall (tpr)')
    ax[0].set_xlabel('1-specificity (fpr)')

    #Annotation Shifts
    scalex = 0.03*(1-0)
    scaley = 0.03*(1-0)

    ax[0].annotate('*',(fm,tm),backgroundcolor='white',fontsize=fnt_sz+10,alpha=0.8) 
    ax[0].annotate('($%.2f \pm %.2f, %.2f \pm %.2f$)'%(fm,fe,tm,te),(fm-scalex,tm-scaley),
                   fontsize=fnt_sz,backgroundcolor='white',alpha=0.8) 

    directory = os.path.dirname(export_fn) #create directory if it doesn't exist
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(export_fn, format='eps', dpi=1000,facecolor=fig.get_facecolor(), edgecolor='none')

def plot_prob_roi(thL,Ptp,Pfp,export_fn='./Figure/roc_savings.eps',parameters={'eta':0.55,'Ci':104,'Ce':14393,'Cne':5840},confidence=1.95):

    Nrows,Nsets = Ptp.shape #number of columns equal number of test sets
    
    fnt_sz=14; lbl_sz=16;
    update_plot_parameters(fnt_sz=fnt_sz,lbl_sz=lbl_sz)
    fig,ax = plt.subplots(1,2,sharex=True,sharey=False,figsize=[15,8])

    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(True)
    cur_axes.axes.get_yaxis().set_visible(True)

    plt.tight_layout(pad=2, w_pad=5, h_pad=0)

    # ROI
    csL,csM,csE = compute_savings(Ptp,Pfp,parameters=parameters)
    j=np.argmax(csM)
    th = thL[j] #threshold for annotation

    for i in range(0,Nsets):
        ax[1].plot(thL,csL[:,i],label='TS%i ($%.2f$, $\$%.0f$)'%(i+1,thL[j],csL[j,i]),
               linestyle='--',color='grey')
    ax[1].plot(thL,csM,label='TSM ($%.2f$, $\$%.0f\pm%.0f$)'%(thL[j],csM[j],csE[j]),
               linestyle='-',color='black')
    ax[1].axvline(th,linestyle=':',color='gray')

    leg = ax[1].legend(loc='lower right')
    ax[1].set_ylabel(r'$E(c_{saving})$')
    ax[1].set_xlabel(r'Threshold ($E(c_i) = \$%.f$, $\eta = %.2f$)'%(parameters['Ci'],parameters['eta']))

    aucL = np.zeros((Nsets,1))
    for i in range(0,Nsets):
        aucL[i]=auc(Pfp[:,i], Ptp[:,i])    
    aucM=auc(Pfp.mean(axis=1), Ptp.mean(axis=1))
    aucE=sem(aucL)*confidence

    # Probabilites
    fm=Pfp.mean(axis=1)[j]
    tm=Ptp.mean(axis=1)[j]
    fe=sem(Pfp,axis=1)[j]*confidence
    te=sem(Ptp,axis=1)[j]*confidence

    for i in range(0,Nsets):
        ax[0].plot(Pfp[:,i],Ptp[:,i],label='TS%i ($%.3f$)'%(i+1,aucL[i]),
                linestyle='--',color='grey')
    ax[0].plot(Pfp.mean(axis=1),Ptp.mean(axis=1),label=r'TSM ($%.2f\pm%.3f$)'%(aucM,aucE),
            linestyle='-',color='black')
    ax[0].legend(loc='lower right')
    ax[0].set_ylabel('$p_{tp}$')
    ax[0].set_xlabel('$p_{fp}$')

    #Annotation Shifts
    scalex = 0.03*(1-0)
    scaley = 0.03*(np.max(Ptp)-np.min(Ptp))

    ax[0].annotate('*',(fm,tm),backgroundcolor='white',fontsize=fnt_sz+10,alpha=0.8) 
    ax[0].annotate('($%.2f \pm %.2f, %.3f \pm %.3f$)'%(fm,fe,tm,te),(fm-scalex,tm-scaley),
                   fontsize=fnt_sz,backgroundcolor='white',alpha=0.8) 

    directory = os.path.dirname(export_fn) #create directory if it doesn't exist
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(export_fn, format='eps', dpi=1000,facecolor=fig.get_facecolor(), edgecolor='none')

def plot_precision_recall(thL,tprL,PrL,export_fn='./Figures/precision_recall.eps',confidence=1.95):

    Nrows,Nsets = tprL.shape #number of columns = number of test sets
    aucL = np.zeros((Nsets,1))
    for i in range(0,Nsets):
        aucL[i]=auc(tprL[:,i], PrL[:,i])  
    aucM=auc(tprL.mean(axis=1), PrL.mean(axis=1))
    aucE=sem(aucL)*confidence

    aucL=np.nan_to_num(aucL)
    aucM=np.nan_to_num(aucM)
    aucE=np.nan_to_num(aucE)

    th = 0.5 #threshold for annotation
    i=np.where(thL==th)
    pm=PrL.mean(axis=1)[i]
    tm=tprL.mean(axis=1)[i]
    pe=sem(PrL,axis=1)[i]*confidence
    te=sem(tprL,axis=1)[i]*confidence

    fnt_sz=14; lbl_sz=16;
    update_plot_parameters(fnt_sz=fnt_sz,lbl_sz=lbl_sz)
    fig,ax = plt.subplots(1,1,sharex=True,sharey=False,figsize=[15,8])
    plt.tight_layout(pad=2, w_pad=5, h_pad=0)

    for i in range(0,Nsets):
        ax.plot(tprL[:,i],PrL[:,i],label='TS%i ($%.2f$)'%(i+1,aucL[i]),
                linestyle='--',color='grey')
    ax.plot(tprL.mean(axis=1),PrL.mean(axis=1),label=r'TSM ($%.2f\pm%.2f$)'%(aucM,aucE),
            linestyle='-',color='black')
    ax.legend(loc='upper right')
    ax.set_ylabel('precision')
    ax.set_xlabel('recall (tpr)')

    #Annotation Shifts
    scalex = 0.03*(1-0)
    scaley = 0.03*(np.max(PrL)-np.min(PrL))

    ax.annotate('*',(tm,pm),backgroundcolor='white',fontsize=fnt_sz+10,alpha=0.8) 
    ax.annotate('($%.2f \pm %.2f, %.2f \pm %.2f$)'%(tm,te,pm,pe),(tm-scalex,pm-scaley),
                   fontsize=fnt_sz,backgroundcolor='white',alpha=0.8) 

    directory = os.path.dirname(export_fn) #create directory if it doesn't exist
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(export_fn, format='eps', dpi=1000,facecolor=fig.get_facecolor(), edgecolor='none')

def compute_savings_variation(thL,Ptp,Pfp,parameters={'Ce':14393,'Cne':5840}):
    # Varies eta or prob. of success parameter from 0 to 1 
    # and intervention costs from $0 to %500 to compute the corresponding
    # maximum mean cost savings (csM_maxL) and confidence interval (csE_maxL)
    # and the threshold at which this occurs (th_maxL)
    etaL = np.arange(0,1.01,0.01)
    ciL = np.arange(0,400,1)

    Nrows = len(ciL)
    Ncols = len(etaL)

    csM_maxL = np.zeros((Nrows,Ncols))
    csE_maxL = np.zeros((Nrows,Ncols))
    th_maxL = np.zeros((Nrows,Ncols))
  
    m = 0
    for c in ciL:
        n = 0
        for e in etaL:
            params = {'eta':e,'Ci':c,'Ce':parameters['Ce'],'Cne':parameters['Cne']}
            csL,csM,csE = compute_savings(Ptp,Pfp,parameters=params)
            j=np.argmax(csM)
            csM_maxL[m,n] = csM[j]
            csE_maxL[m,n] = csE[j]
            th_maxL[m,n] = thL[j]
            n+=1
        m+=1
    return etaL,ciL,th_maxL,csM_maxL,csE_maxL


def plot_parameter_space(thL,Ptp,Pfp,export_fn='./Figures/parameter_space.eps',parameters={'Ce':14393,'Cne':5840}):

    etaL,ciL,th_maxL,csM_maxL,csE_maxL = compute_savings_variation(thL,Ptp,Pfp,parameters=parameters)
    mineta=np.min(etaL)
    maxeta=np.max(etaL)
    minci=np.min(ciL)
    maxci=np.max(ciL)

    fnt_sz=14; lbl_sz=16;
    update_plot_parameters(fnt_sz=fnt_sz,lbl_sz=lbl_sz)
    fig,ax = plt.subplots(1,3,sharex=True,sharey=False,figsize=[15,5])
    plt.tight_layout(pad=2, w_pad=5, h_pad=0)
    
    c=ax[0].imshow(th_maxL, cmap='gist_stern', interpolation='none', extent=[mineta,maxeta,maxci,minci], aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="10%", pad=0.05)
    clb=plt.colorbar(c, cax=cax)
    clb.set_label('Threshold',rotation=270,labelpad=20)
    ax[0].set_ylabel(r'$E(c_i)$')
    ax[0].set_xlabel(r'$\eta$')

    c=ax[1].imshow(csM_maxL, cmap='gist_stern', interpolation='none', extent=[mineta,maxeta,maxci,minci], aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="10%", pad=0.05)
    clb=plt.colorbar(c, cax=cax)
    clb.set_label(r'$E(c_s)$',rotation=270,labelpad=20)
    #ax[1].set_ylabel('E(c_i)')
    ax[1].set_xlabel(r'$\eta$')

    c=ax[2].imshow(csE_maxL, cmap='gist_stern', interpolation='none', extent=[mineta,maxeta,maxci,minci], aspect='auto')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="10%", pad=0.05)
    clb=plt.colorbar(c, cax=cax)
    clb.set_label(r'$\pm$ 95\% CI',rotation=270,labelpad=20)
    #ax[2].set_ylabel('E(c_i)')
    ax[2].set_xlabel(r'$\eta$')

    directory = os.path.dirname(export_fn) #create directory if it doesn't exist
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(export_fn, format='eps', dpi=1000,facecolor=fig.get_facecolor(), edgecolor='none')

def get_intervention_data():
    ci_otago = 339.15
    eta_otago = 0.35
    ci_tai_chi = 104.02
    eta_tai_chi = 0.55
    ci_stepping_on = 211.38
    eta_stepping_on = .31
    eta_ci = {'Otago Exercise':(eta_otago,ci_otago),
            'Tai Chi':(eta_tai_chi,ci_tai_chi),
            'Stepping On':(eta_stepping_on,ci_stepping_on)}
    return eta_ci

def plot_parameter_space_contour(thL,Ptp,Pfp,export_fn='./Figures/parameter_space_contour.eps',parameters={'Ce':14393,'Cne':5840}):

    etaL,ciL,th_maxL,csM_maxL,csE_maxL = compute_savings_variation(thL,Ptp,Pfp,parameters=parameters)
    mineta=np.min(etaL)
    maxeta=np.max(etaL)
    minci=np.min(ciL)
    maxci=np.max(ciL)

    ETA,CI = np.meshgrid(etaL,ciL)

    fnt_sz=14; lbl_sz=16; fmt = r'$\$%.f$'
    update_plot_parameters(fnt_sz=fnt_sz,lbl_sz=lbl_sz)
    fig,ax = plt.subplots(1,3,sharex=True,sharey=False,figsize=[15,8])
    plt.tight_layout(pad=2, w_pad=5, h_pad=0)
    
    levels = [0,10,20,40,60,80,100]
    c=ax[0].contour(ETA,CI,csM_maxL-csE_maxL,interpolation='none',levels=levels,
        extent=[mineta,maxeta,maxci,minci],aspect='auto',colors='black')
    print(c.locate_label)
    ax[0].clabel(c,inline=1, fmt=fmt)
    ax[0].set_ylabel(r'$E(c_i)$')
    ax[0].set_xlabel(r'$\eta$')
    ax[0].set_title(r'$E(c_s)$ - 95\% CI')

    c=ax[1].contour(ETA,CI,csM_maxL,interpolation='none',levels=levels,
        extent=[mineta,maxeta,maxci,minci],aspect='auto',colors='black')
    ax[1].clabel(c,inline=1,fmt=fmt)
    ax[1].set_ylabel(r'$E(c_i)$')
    ax[1].set_xlabel(r'$\eta$')
    ax[1].set_title(r'$E(c_s)$')

    c=ax[2].contour(ETA,CI,csM_maxL+csE_maxL,interpolation='none',levels=levels,
        extent=[mineta,maxeta,maxci,minci],aspect='auto',colors='black')
    ax[2].clabel(c,inline=1,fmt=fmt)
    ax[2].set_ylabel(r'$E(c_i)$')
    ax[2].set_xlabel(r'$\eta$')
    ax[2].set_title('$E(c_s)$ + 95\% CI')

    #Annotation Shifts
    scalex = 0.03*(maxeta-mineta)
    scaley = 0.03*(maxci-minci)

    eta_ci = get_intervention_data()
    for key,value in eta_ci.items(): 
        for i in (0,1,2):
            (x,y)=value 
            ax[i].annotate('*',(x,y),backgroundcolor='white',fontsize=fnt_sz+10,alpha=0.4)
            ax[i].annotate(r'\textit{'+key+'}',(x-scalex,y-scaley),
                           fontsize=fnt_sz,backgroundcolor='white',alpha=0.8) 
    
    directory = os.path.dirname(export_fn) #create directory if it doesn't exist
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(export_fn, format='eps',dpi=1000,facecolor=fig.get_facecolor(), edgecolor='none')

def plot_ols_fit(y_pred,y_true,label=['Predicted','Actual'],formula='',table='',export_fn='./Figures/ols_fit.png'):
    # Plots actual vs predicted points
    # Input is numpy array
    rsq = compute_rsq(y_pred,y_true)
    miny = np.min([y_pred,y_true])
    maxy = np.max([y_pred,y_true])
    pady = (maxy-miny)*0.1

    fnt_sz=14; lbl_sz=16;
    update_plot_parameters(fnt_sz=fnt_sz,lbl_sz=lbl_sz)
    fig,ax = plt.subplots(1,1,sharex=False,sharey=False,figsize=[10,6])

    ax.plot(y_pred,y_true,'.')
    ax.set_xlabel('Predicted Weight Change (kg)')
    ax.set_ylabel('Actual Weight Change (kg)')
    ax.set_ylim([miny-pady,maxy+pady])
    ax.set_xlim([miny-pady,maxy+pady])
    #ax.set_aspect('equal')

    # Plot first-order polynomial to data
    fit = np.polyfit(y_pred, y_true, deg=1)
    x = np.arange(miny,maxy)
    ax.plot(x, fit[0] * x + fit[1],label='actual',
        color='lightblue') 
    
    # Find limits for expected regression line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # Plot expected regression line
    ax.plot(lims, lims,label='predicted',
        color='lightgray', alpha=0.75, zorder=0)
    
    ax.legend(title=r'Trend ($R^2$=%.3f)'%round(rsq,3),loc='lower right')

    # # Add R^2 value:
    # pos = [(lims[1]-lims[0])*0.8+lims[0],(lims[1]-lims[0])*0.1+lims[0]]
    # ax.annotate('$R^2$=%.2f'%rsq,pos,
    #             backgroundcolor='white',fontsize=fnt_sz,alpha=0.8) 

    pos = [(lims[1]-lims[0])*1.05+lims[0],(lims[1]-lims[0])*0.75+lims[0]]
    ax.text(pos[0],pos[1],formula)#,
                #backgroundcolor='white',fontsize=fnt_sz,alpha=0.8) 

    pos = [(lims[1]-lims[0])*1.1+lims[0],(lims[1]-lims[0])*0.07+lims[0]]
    ax.text(pos[0],pos[1],table,
                 backgroundcolor='white',fontsize=fnt_sz,alpha=0.8) 

    directory = os.path.dirname(export_fn) #create directory if it doesn't exist
    if not os.path.exists(directory):
            os.makedirs(directory)
    plt.savefig(export_fn, format='png',dpi=500,
        facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')

def param2latex(res):
    # This creates a latex table using statsmodels regression results object
    coef = res.params.values.tolist() #coefficients
    pval = res.pvalues.values.tolist() #their p-values
    #95% confidence interval of coefficeints
    ci0 = res.conf_int()[0].tolist()
    ci1 = res.conf_int()[1].tolist()
    ci = [r'%.2f \ \ %.2f'%(ci0[i],ci1[i]) for i in range(0,len(ci0))]
    # coef labels
    clabel = []
    for i in range(0,len(coef)):
        clabel.append(r'c_%i'%i)
    # create latex table
    headers = (r'coef',r'[95.0\% CI]',r'p-value')
    table_latex = r"\begin{tabular}{lrrr}"
    table_latex += r' & %s & %s & %s '%headers
    table_latex += r'\\ \hline '
    for i in range(0,len(pval)):
        if pval[i]<0.05: #significant
            table_latex += r'\\ $%s*$ & %.2f & %s & %.2f'%(clabel[i],coef[i],ci[i],pval[i])
        else: 
            table_latex += r'\\ $%s$ & %.2f & %s & %.2f'%(clabel[i],coef[i],ci[i],pval[i])
    table_latex += r'\\ \hline '
    table_latex += r'\multicolumn{4}{l}{$*$ statistically significant ($p<0.05$)}'
    table_latex += r'\end{tabular}'
    return table_latex

