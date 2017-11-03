# MVC framework for Predictive Models
# Author: Rupinder Singh

try:
    reload
except NameError:
    # Python 3
    from importlib import reload
import numpy as np
import sys
import os
import imp
import time 
import pickle
from tabulate import tabulate

#IMPORTANT: Supply pLib directory and Export Directory
cfg_pth = '/data/perf/scripts/analytics'
pth_lib = os.path.join(cfg_pth, 'pLib/') 

# IMPORT CUSTOM MODULES
rds = imp.load_source('rs_desc_stats', pth_lib+'rs_desc_stats.py')
rmm = imp.load_source('rs_model_metrics', pth_lib+'rs_model_metrics.py')
rmp = imp.load_source('rs_model_pipeline', pth_lib+'rs_model_pipeline.py')
rsql = imp.load_source('rs_mysql', pth_lib+'rs_mysql.py')

from sklearn import metrics

class baseModel:
    def __init__(self):
        #Initialize this class when called by subclass 
        self.event = False #name of event/target/label
        self.feature_type = False #labels for type of features
        self.trained_scaler = False #scaler used to transform data to apporpriate scale
        self.lr_model = False #logistic regression model
        self.X_train = False #data set features used for training
        self.y_train = False #data set labels used for training

    def loadData(self):
        if self.event is False:
            raise NotImplementedError("Subclass must implement abstract method")

    def scaleFeatures(self, X, trained_scaler = False, feature_type = False):
        print("Scaling Features...")
        t1 = time.time()
        if trained_scaler is False:
            unq_vals, self.feature_type = rmp.type_of_feature(X, print_flag = False)
            Xs, self.trained_scaler = rmp.auto_scale_features(X, self.feature_type)
        else:
            Xs, trained_scaler = rmp.auto_scale_features(X, feature_type, scaler = trained_scaler)
        t_log_cv = time.time() - t1
        print("Done! Time elapsed: {0:.2f} seconds".format(t_log_cv))
        return Xs

    def calibrateModel(self, X, y):
        print("Fitting Data...")
        t1 = time.time()
        self.lr_model = rmp.logistic_regression_ridge(X, y)
        t_log_cv = time.time() - t1
        print("Done! Time elapsed: {0:.2f} seconds".format(t_log_cv) )

    def getEventProb(self, X):
        return rmp.predict_score(self.lr_model, X)

class Model(baseModel):
    def __init__(self):
        pass

    def loadData(self, dataParams, pckl_fn = False):
        if pckl_fn:
            df = rsql.get_dataframe('', host_in = 'client', pkl_fn = pckl_fn)
        else:
            sql = "enter mysql query that grabs features"
            print(sql)
            df = rsql.get_dataframe(sql, host_in='client')
            df = df.convert_objects(convert_numeric = True) #needed to convert some objects to ints/floats
            try:
                df.drop(['Null'], 1)
            except ValueError:
                None
            df.fillna(0, inplace = True)
        return df

class EDModel(Model):
    def __init__(self):
        self.event = 'ed_target' #create ed_nxtYr column

class ReadmissionModel(Model):
    def __init__(self):
        self.event = 'lt60d_readmit' #create lt60d_readmit_nxtYr column

class HospitalModel(Model):
    def __init__(self):
        self.event = 'hospitalized' #create hospitalized_nxtYr column

class FallModel(Model):
    def __init__(self):
        self.event = 'traumatic_fall' #create traumatic_fall_nxtYr column

class View:
    def __init__(self):
        pass

    def printList(self, lst):
        print(lst)

    def printTable(self, tbl):
        print(tabulate(tbl))

    def printEventStats(self, stats_by_event, stats_by_event_by_yr):
        #self.printFull(stats_by_event)
        print(stats_by_event)
        print("---------------------------------------------------------------------------------------------------")
        #self.printFull(stats_by_event_by_yr)
        print(stats_by_event_by_yr)

class Controller:
    def __init__(self, model_name, train_model, unpkl_data, data_params):
        self.initModel(model_name) #classifier model (self.model)
        self.train_model = train_model # 0: use existing model (pickle) to make predictions, 1: train model, 2: test model (use existing model from pkl)
        self.unpkl_data = unpkl_data # 0: load data from sql, 1: load data from pickle
        self.data_params = data_params #data parameters for sql query
        self.pth_file = os.path.dirname(os.path.abspath(__file__)) #path to this python file
        self.pth_data = os.path.join(self.pth_file, 'Data/') #path to data
        self.pth_figures = os.path.join(self.pth_file, 'Figures/') #path to figures
        self.fn_model = os.path.join(self.pth_data, 'ultimate_'+model_name+'_model.pkl') #location of model
        self.fn_data = os.path.join(self.pth_data, 'df_ultimate_'+model_name+'.pkl') #location of data used for training
        self.fn_coef = os.path.join(self.pth_data, model_name+'_model_coef.csv') #location of coefficient file
        self.cases = {1:'th_metrics', 2:'roc_roi', 3:'prob_roi', 4:'prec_recall', 5:'param_space', 6:'param_contour'}
        if train_model == 1:
            self.fn_roc = os.path.join(self.pth_data, "overall_" + model_name + "_roc.csv") #location of roc data
            self.fn_plot = lambda i: os.path.join(self.pth_figures, "overall_" + model_name + "_" + self.cases[i]+'.eps')
        else:
            self.fn_roc = os.path.join(self.pth_data, str(self.data_params['srvcYr']) + '_' + self.data_params['pyrID'] + "_" + model_name + "_roc.csv") 
            self.fn_plot = lambda i: os.path.join(self.pth_figures, str(self.data_params['srvcYr']) + '_' + self.data_params['pyrID'] + "_" + model_name + "_" + self.cases[i]+'.eps')
        self.fn_predict = os.path.join(self.pth_data, self.data_params['pyrID']+"_"+model_name+"_risk_predict.csv") #location of model predictions
        self.n_test_sets = 4 #number of subsets created from test set
        #params: eta(intervention effictiveness -> 0-1), Ci(intervention cost), Ce(event cost), Cne(no event cost)
        self.cost_params = {'eta':0.55,'Ci':104,'Ce':15034,'Cne':5930} #Update these for your target of interest!!!

    def initModel(self, model_name):
        try:
            del self.model
        except:
            pass
        modelClass = {'fall':FallModel, 
                      'hosp':HospitalModel, 
                      'readm':ReadmissionModel, 
                      'ed':EDModel}
        self.model = modelClass[model_name]() #create specific model class instance

    def loadData(self, scale_data = False):
        if self.unpkl_data == 1:
            self.df = self.model.loadData(self.data_params, pckl_fn = self.fn_data)
        else:
            self.df = self.model.loadData(self.data_params)
        self.patients = self.df['patient_id']
        self.y = self.df[self.model.event]
        self.X = self.df.drop([self.model.event,'index','patient_id','service_year'], 1)
        if scale_data:
            self.X = self.model.scaleFeatures(self.X, trained_scaler = self.model.trained_scaler, feature_type = self.model.feature_type)

    def saveData(self):
        rsql.save_dataframe(self.df, self.fn_data)

    def loadModel(self):
        if self.model is False:
            raise NotImplementedError("Model class must be initialized")
        with open(self.fn_model, 'rb') as pkl:
            model_old = pickle.load(pkl)
        self.model.trained_scaler = model_old.trained_scaler
        self.model.feature_type = model_old.feature_type
        self.model.lr_model = model_old.lr_model
        self.model.X_train = model_old.X_train
        self.model.y_train = model_old.y_train
    
    def buildModel(self):
        # Build Model if prexisting one doesn't exist
        self.model.X_train, self.X_test, self.model.y_train, self.y_test = rmp.split_data(self.X, self.y, test_size=.3,random_state=0)
        #run scaleFeatures on training dataset first so model.trained_scaler can be initialized
        self.model.X_train = self.model.scaleFeatures(self.model.X_train, trained_scaler = False, feature_type = False)
        self.X_test = self.model.scaleFeatures(self.X_test, trained_scaler = self.model.trained_scaler, feature_type = self.model.feature_type)
        self.X = self.model.scaleFeatures(self.X, trained_scaler = self.model.trained_scaler, feature_type = self.model.feature_type)
        self.model.calibrateModel(self.model.X_train, self.model.y_train)

    def saveModel(self):
        directory = os.path.dirname(self.fn_model)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.fn_model, 'wb') as pickle_file:
            pickle.dump(self.model, pickle_file)

    def getClassificationReport(self):
        score = self.getClassificationScore(print_flag = True)
        if self.train_model == 1:
            coef_list = self.getCoef(print_flag = True)
            event_incidence = self.getEventIncidence(print_flag = True)
            stats_by_event_by_yr, stats_by_event = self.getEventStats(print_flag = True)
        roc_data = self.getROC()
        self.getMetrics()
        for i, case in self.cases.iteritems():
            self.plotMetrics(i)

    def makePredictions(self):
        self.y_prob = self.model.getEventProb(self.X)
        self.predict_result = np.concatenate((np.transpose(np.matrix(self.patients)), self.y_prob), axis = 1)
        np.savetxt(self.fn_predict, self.predict_result, fmt = ['%s','%2.3f','%2.3f'], delimiter=",", header = "patient_id, proabability_zero, proabability_one")

    def runModel(self):
        if self.train_model == 0: #Make Predictions
            self.loadModel()
            self.loadData(scale_data = True)
            self.makePredictions()
        elif self.train_model == 1: #Train Model
            self.loadData(scale_data = False) #scaling will happen during model build
            self.saveData()
            self.buildModel()
            self.getClassificationReport()
            self.saveModel()
        else: #Test Existing Model
            self.loadModel()
            self.loadData(scale_data = True)
            self.X_test = self.X
            self.y_test = self.y
            self.getClassificationReport()

    def getCoef(self, print_flag = False):
        coef_lst = rmm.get_model_coef(self.model.lr_model, self.X_test.columns, export_fn=self.fn_coef)
        if print_flag:
            view = View()
            view.printList(coef_lst)
        return coef_lst

    def getClassificationScore(self, print_flag = False):
        score = rmm.compute_classification_score(self.model.lr_model, self.model.y_train, self.model.X_train, self.y_test, self.X_test, self.model.event)
        if print_flag:
            view = View()
            view.printTable(score)
        return score

    def getROC(self):
        self.test_size = self.X_test.shape[0]
        self.coded_size = np.count_nonzero(self.y_test[:]==1)
        y_test_prob = self.model.getEventProb(self.X_test)
        roc_fpr, roc_tpr, roc_threshold = metrics.roc_curve(np.array(self.y_test), np.array(np.matrix(y_test_prob)[:,1]), pos_label = 1)
        final_roc_data = np.transpose(np.concatenate((np.matrix(roc_fpr),np.matrix(roc_tpr),np.matrix(roc_threshold)),axis=0))
        final_roc_data = np.insert(final_roc_data, final_roc_data.shape[1], self.test_size, axis=1)
        final_roc_data = np.insert(final_roc_data, final_roc_data.shape[1], metrics.auc(roc_fpr, roc_tpr), axis=1)
        final_roc_data = np.insert(final_roc_data, final_roc_data.shape[1], self.coded_size, axis=1)
        np.savetxt(self.fn_roc, final_roc_data, fmt = ['%s','%s','%s','%s','%s','%s'], delimiter=",")
        return final_roc_data

    def getEventIncidence(self, print_flag = False):
        incidence_tbl = rds.event_incidence(self.df, self.model.event)
        if print_flag:
            view = View()
            view.printTable(incidence_tbl)
        return incidence_tbl

    def getEventStats(self, print_flag = False): 
        stats_by_event_by_yr,stats_by_event = rds.analyze_event(self.df[[self.model.event, 'service_year', 'risk_score', 'total_cost']])
        if print_flag:
            print('stats_by_event_by_yr::',stats_by_event_by_yr)
            print('stats_by_event::',stats_by_event)
            view = View() 
            view.printEventStats(stats_by_event, stats_by_event_by_yr)
        return stats_by_event_by_yr, stats_by_event

    def getMetrics(self):
        self.thL, self.tnrL, self.fprL, self.fnrL, self.tprL, self.PrL, self.Ptp, self.Pfp, self.Ptn, self.Pfn = \
            rmm.model_metrics(self.model.lr_model, self.y_test, self.X_test, self.n_test_sets)

    def plotMetrics(self, case):
        if self.cases[case] == 'th_metrics': 
            rmm.plot_model_metrics(self.thL, self.fprL, self.tprL, self.PrL, export_fn = self.fn_plot(case))
        if self.cases[case] == 'roc_roi':
            rmm.plot_roc_roi(self.thL, self.fprL, self.tprL, self.Ptp, self.Pfp, export_fn = self.fn_plot(case), parameters = self.cost_params)
        if self.cases[case] == 'prob_roi':
            rmm.plot_prob_roi(self.thL, self.Ptp, self.Pfp, export_fn = self.fn_plot(case), parameters = self.cost_params)
        if self.cases[case] == 'prec_recall':
            rmm.plot_precision_recall(self.thL, self.tprL, self.PrL, export_fn = self.fn_plot(case))
        if self.cases[case] == 'param_space':
            rmm.plot_parameter_space(self.thL, self.Ptp, self.Pfp, export_fn = self.fn_plot(case))
        if self.cases[case] == 'param_contour':
            rmm.plot_parameter_space_contour(self.thL, self.Ptp, self.Pfp, export_fn = self.fn_plot(case))

if __name__ == '__main__':
# Supplied Arguments (argv): train_model?, unpkl_data, pyrID, srvcYr, runFall?, runHosp?, runReadm?, runED?  
    print("Running %s ...."%(sys.argv[0]))
    t1 = time.time()
    train_model = int(sys.argv[1]) # 0: use existing model (pickle) to make predictions, 1: train model, 2: test model using model pickle
    unpkl_data = int(sys.argv[2]) # 0: load data from sql, 1: load data from pickle 
    data_params = {'pyrID' : sys.argv[3],
                    'srvcYr' : int(sys.argv[4])}
    #Flags to determine which models to evaluate/compute/generate score for
    run_model = {'fall':int(sys.argv[5]), 
                'hosp':int(sys.argv[6]), 
                'readm':int(sys.argv[7]), 
                'ed':int(sys.argv[8])}
    """Sequentially invoke the models one by one"""
    for model, runm in run_model.iteritems():
        if runm == 1:
            ctrl = Controller(model, train_model, unpkl_data, data_params)
            ctrl.runModel()
    t_log_cv = (time.time() - t1)/60.
    print("Done! Time elapsed: {0:.2f} minutes".format(t_log_cv) )

