# Custom module to streamline feature transformations
# Author: Rupinder Singh (Oct. 25, 2016)

from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm
import statsmodels.formula.api as smf

def drop_empty_features(df):
    #this function drops features from a pandas dataframe that have only a single value
    unq_fetr_vals = np.array([len(df.ix[:,n].unique()) for n in range(0,len(df.columns))])
    idx = np.where(unq_fetr_vals==1) 
    for i in idx:
        df.drop(df.columns[i], axis=1)

def minmax_scaler(X):
    # scale from 0 to 1
    xmin =  X.min(axis=0)
    Xt = (X - xmin) / (X.max(axis=0) - xmin)
    return Xt

def minmax_scaler(X,scaler=False):
    # X: nxm array. 
    # Scaler:    ~False - X is transformed using input scaler. 
    #            False - X is used to train scaler and then transformed.
    # scaler: 
    # Xt: Transformed data. Axis=0 of array is scaled from 0 to 1. 
    if scaler is False:
        scaler = preprocessing.MinMaxScaler()
        scaler = scaler.fit(X)
    Xt = scaler.transform(X)
    return Xt,scaler

def normalizer(X,scaler=False,norm='l2'): #! this doesn't normalize along axis=0
    # Divide features by their l2/l1 norms
    if scaler is False:
        scaler = preprocessing.Normalizer(norm=norm)
        scaler = scaler.fit(X)#this does nothing
    Xt = scaler.transform(X)
    return Xt,scaler

def standardizer(X,scaler=False):
    # Scale data to have zero mean and unit variance
    if scaler is False:
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(X)
    Xt = scaler.transform(X)
    return Xt,scaler

def binarizer(X,scaler=False,threshold=0.5):
    # Feature binarization 
    if scaler is False:
        scaler = preprocessing.Binarizer(threshold=threshold)
        scaler = scaler.fit(X)
    Xt = scaler.transform(X)
    return Xt,scaler

def robust_standardizer(X,scaler=False):
    #Alternative to standardizer. Standardizes using median and IQR instead of mean and variance
    if scaler is False:
        scaler = preprocessing.RobustScaler()
        scaler = scaler.fit(X)
    Xt = scaler.transform(X)
    return Xt,scaler

def poly_features(X,poly=False,degree=2,interaction_only=False):
    #Add polynomial features
    if poly is False:
        poly = preprocessing.PolynomialFeatures(degree=degree,interaction_only=interaction_only)
        poly = poly.fit(X) # this does nothing
    Xt = poly.transform(X)
    return Xt,poly

def log1p_tansform(X):
    transformer = preprocessing.FunctionTransformer(np.log1p)
    Xt = transformer.transform(X)
    return Xt

def type_of_feature(X,print_flag=True):
    # X here is assumed to be pandas dataframe
    # Find length of each feature and output index for binary, continuous and singleton
    uniq_vals = np.array([len(X.ix[:,n].unique()) for n in range(0,len(X.columns))])
    sin_clms = np.where(uniq_vals==1)[0]
    bin_clms = np.where(uniq_vals==2)[0]
    cont_clms = np.where(uniq_vals>2)[0]
    features = X.columns.values.T

    uniq_vals = pd.DataFrame(np.vstack((features,uniq_vals)).T,
                            columns=['features','unique_values'])
    sin_clms = pd.DataFrame(np.vstack((sin_clms,features[sin_clms])).T,
                            columns=['sin_clm','singleton_feature'])
    bin_clms = pd.DataFrame(np.vstack((bin_clms,features[bin_clms])).T,
                            columns=['bin_clm','binary_feature'])
    cont_clms = pd.DataFrame(np.vstack((cont_clms,features[cont_clms])).T,
                            columns=['cont_clm','continuous_feature'])

    if print_flag:
        print('Singleton Features:')
        print(sin_clms.values)
        print('\nBinary Features:')
        print(bin_clms.values)
        print('\nCont. Features:')
        print(cont_clms.values)

    return uniq_vals,(sin_clms,bin_clms,cont_clms)

def auto_scale_features(X,feature_type,scaler=False):
    """
    # X: A pandas dataframe with column-wise features
    # feature_type: tuple of dataframes of features that are (singleton,binary,continuous)
    # sin,bin,cont are assumed to be pandas dataframe with first column having column id
    #     for singleton, binary, continuous features
    # Output:     Binary features  are scaled from 0 to 1;
    #             Continuous features are standardized
    #             Singleton features are dropped
    """
    sin_clms = feature_type[0]
    bin_clms = feature_type[1]
    cont_clms = feature_type[2]
    
    sinj = sin_clms.values[:,1].tolist() #labels
    binj = bin_clms.values[:,0].tolist() #indexes
    contj = cont_clms.values[:,0].tolist() #indexes

    if scaler is False:
        X.ix[:,binj],bin_scaler = minmax_scaler(X.ix[:,binj])
        X.ix[:,contj],cont_scaler = standardizer(X.ix[:,contj])
    else:
        X.ix[:,binj],bin_scaler = minmax_scaler(X.ix[:,binj],scaler=scaler[0])
        X.ix[:,contj],cont_scaler = standardizer(X.ix[:,contj],scaler=scaler[1])
    

    X = X.drop(sinj,1)

    return X,(bin_scaler,cont_scaler)

def split_data(X,y,test_size=0.3,random_state=0):
    # Using sklearn api
    return train_test_split(X, y, test_size=.3,random_state=0)

def logistic_regression_lasso(X,y):
    # Function fits logit(P(y)=1) ~ Beta*X
    # with l1 regularization (lasso)
    # X and y are assumed to be pandas dataframes and series
    # Output is sklearn model
    Cs = 1 #inverse_regularization_strength 1e-4 to 1e4
    folds = 2 #numker of folds used
    penalty = 'l1' #use lasso
    #from sklearn.metrics import SCORERS
    #print(SCORERS.keys()) #possible values for scoring
    #coring = "precision" #compute precision of class labeled as 1
    #scoring = "recall"
    scoring = "accuracy"
    n_jobs = 1 #use all cores
    solver = 'liblinear'
    model = LogisticRegressionCV(Cs=Cs, fit_intercept=True, class_weight='balanced', cv=folds, 
        penalty=penalty, n_jobs=n_jobs, scoring=scoring, solver=solver)
    model.fit(X.values,y.values)
    return model

def logistic_regression_ridge(X,y):
    # Function fits logit(P(y)=1) ~ Beta*X
    # with l2 regularization (ridge)
    # X and y are assumed to be pandas dataframes and series
    # Output is sklearn model
    Cs = 1 #inverse_regularization_strength 1e-4 to 1e4
    folds = 2 #numker of folds used
    penalty = 'l2' #use ridge
    #from sklearn.metrics import SCORERS
    #print(SCORERS.keys()) #possible values for scoring
    #coring = "precision" #compute precision of class labeled as 1
    #scoring = "recall"
    scoring = "accuracy"
    n_jobs = 1 #use all cores
    model = LogisticRegressionCV(Cs=Cs, fit_intercept=True, class_weight='balanced', cv=folds, 
        penalty=penalty, n_jobs=n_jobs, scoring=scoring)
    model.fit(X.values,y.values)
    return model

def logistic_regression(df,formula=False,X=False,y=False):
    # This uses statsmodel api to fit data via logistic regression
    # df: is the dataframe
    # formula: is a function (e.g., 'traumatic_fall ~ age + age_group + Stroke')
    # if formula is not defined then, X, y are needed
    # X: dataframe (with constant column already added e.g. X=sm.add_constant(X)) 
    if formula:
        res = smf.logit(formula = func, data = df).fit()
    else:
        res = sm.Logit(y,X.ix[:,:]).fit()
        #res = sm.Logit(y,X.ix[:,:]).fit_regularized(alpha=10)
    #print(res.summary())
    #str_latex=res.summary().as_latex()
    #with open('table.tex', 'w') as file_:
    #    file_.write(str_latex)
    return res1

def predict_score(model,X):
    # X is pandas dataframe
    # model is sklearn model object
    return model.predict_proba(X)
