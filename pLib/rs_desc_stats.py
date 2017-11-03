# Custom module to extract descriptive stats from dataframe
# Author: Rupinder Singh (Oct. 25, 2016)

from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import sem,ttest_ind
import re
from collections import Counter

def event_incidence(df,event=False):
	# This function computes event incidence
	# df: pandas dataframe
	# event: column name to compute incidence on
	event_count = len(df[df[event]==1])
	no_event_count = len(df[df[event]==0])
	tot = event_count + no_event_count
	event_prev = event_count/tot*100
	no_event_prev = no_event_count/tot*100
	tbl=[['', event, 'no_'+event, 'total'],
    ['count', event_count, no_event_count, tot],
    ['incidence', '{0:.2f}%'.format(event_prev), '{0:.2f}%'.format(no_event_prev), '100%']]
	return tbl

def analyze_event(df):
	# This function will compute stats by event
	# df: data frame with following format df[['event','service_year','risk_score','cost']]
	# rdf: contains stats by event and service year
	# rdf2: contains stats mean and sem by event
	
	# Get Column Names
	clm_event = df.columns[0]
	clm_yr = df.columns[1]
	clm_risk = df.columns[2]
	clm_cost = df.columns[3]
	clm_cost_ra = clm_cost+'_risk_adjusted'

	# Compute risk-adjusted cost
	df[clm_cost_ra] = df[clm_cost]/df[clm_risk]

	# Compute stats on risk and cost
	rdf = df.groupby([clm_event,clm_yr]).agg(['sum','count','mean']).reset_index()
	rdf = merge_multilevels(rdf) #remove multilevels
	
	# Compute Total Member Count by Year
	df_count = df[[clm_yr,clm_event]].groupby([clm_yr]).agg(['count']).reset_index()
	df_count = merge_multilevels(df_count)
	clm_member_count_by_yr = 'member_count_by_year'
	df_count.columns = [clm_yr,clm_member_count_by_yr]
	rdf = rdf.merge(df_count,how='inner',on=clm_yr)

	# Compute Incidence of event
	clm_incidence = 'event_incidence'
	clm_member_count = 'member_count'
	rdf = rdf.rename(columns={clm_cost+'_count':clm_member_count})
	rdf[clm_incidence] = rdf[clm_member_count]/rdf[clm_member_count_by_yr]


	# Compute total risk_adjusted cost by year
	df_cost = df[[clm_yr,clm_cost_ra]].groupby([clm_yr]).agg(['sum']).reset_index()
	df_cost = merge_multilevels(df_cost)
	clm_cost_ra_by_yr = 'cost_risk_adjusted_by_year'
	df_cost.columns = [clm_yr,clm_cost_ra_by_yr]
	rdf = rdf.merge(df_cost,how='inner',on=clm_yr)

	# Compute risk-adjusted cost fraction
	clm_cost_ra_fraq = clm_cost_ra+'_fraction'
	rdf[clm_cost_ra_fraq] = rdf[clm_cost_ra+'_sum']/rdf[clm_cost_ra_by_yr]
	
	rdf = rdf.drop([clm_risk+'_count',clm_cost_ra+'_count'],axis=1)

	rdf2 = rdf.groupby(clm_event).agg(['mean',sem]).reset_index()

	rdf2.pop(clm_yr)
	rdf2.pop(clm_cost_ra_by_yr)
	rdf2.pop(clm_member_count_by_yr) #remove clm_yr

	return rdf,rdf2

def successive_diff(df):
	return 1

def merge_multilevels(df,remove_spaces=False):
	if remove_spaces:
		df.columns = [' '.join(col).strip().replace(' ','_') for col in df.columns.values]
	else:
		df.columns = [re.sub(r'_$','','_'.join(col).strip()) for col in df.columns.values]
	return df

def replace_spaces(str1):
	str1 = ' '.join(str1).strip().replace(' ','_')
	return str1

def describe(df):
	# Describes Data Frame
	nrws, nclms = df.shape
	clms = df.columns
	print('Basic Stats')
	print('-----------------')
	print("  Shape: %i x %i"%(nrws,nclms))
	print("  Columns: %s"%(clms.values))

	# Compute overview dictionary of unique, repeated, and null entries
	rdic = {}
	for clm in clms:
		rdic[clm] = {'Null':{},'Unique':{},'Repeated':{}}

		# Nulls
		nul = sorted(df[df[clm].isnull()].index)
		rdic[clm]['Null']['index'] = nul
		rdic[clm]['Null']['count'] = len(nul)
		# Unique
		unq = sorted(df[clm].unique(), 
				key=lambda x: (x is not np.nan, x))	
		rdic[clm]['Unique']['list'] = unq
		rdic[clm]['Unique']['count'] = len(unq)
		# Repeated
		rpt = sorted([item for item, count in Counter(df[clm].values).items() if count > 1],
				key=lambda x: (x is not np.nan, x))
		rdic[clm]['Repeated']['list'] = rpt
		rdic[clm]['Repeated']['count'] = len(rpt)
		df[clm].isnull()

	# Reform dictionary to format appropriate to create multiindex dataframe
	rdic2 = reform_dict(rdic)

	rdf = pd.DataFrame(rdic2)
	rdf.index = pd.MultiIndex.from_tuples(rdf.index) 

	return rdf

def reform_dict(rdic):
	# Reforms a 3 level dictionary to a 2 level dictionary. 
	# Level 1, Level 2, Level 3 -> Level 1, Level (2,3)
	rdic2={}
	for k0,v0 in rdic.items():
		rdic2[k0] = {}
		for k1,v1 in v0.items():
			for k2,v2 in v1.items():
				rdic2[k0][(k1,k2)] = v2
	return rdic2

def welch_test(df1,df2,alpha=0.05):
	# Create table of results from two independent sample unequal variance t-test
	# Input are dataframes.
	rtp = []
	for column in df1:
		m1 = df1[column].mean()
		m2 = df2[column].mean()
		t,p = ttest_ind(df1[column],df2[column], equal_var=False)
		if p<0.05:
			rtp.append([column,m1,m2,m1-m2,t,p,'yes'])
		else:
			rtp.append([column,m1,m2,m1-m2,t,p,''])
	rtp = pd.DataFrame(rtp,columns=['variable','treatment','control','diff','t statistic','p-value','reject H0'])
	return rtp