# Custom module to connect to mysql and retrieve data
# Author: Rupinder Singh (Oct. 25, 2016)

# MySQL Connectors
from sqlalchemy import create_engine as sqlengine, text as sqltext
import pickle
import pandas as pd
import os
from psutil import cpu_times_percent
import subprocess as sp

def load_config_file(cfgFile):
    """ Parsed content of Config file into Dictionary 
        [client] ---> becomes primary-key with values as what follows
        host="host"
        user="un" ---> becomes secondary key-values
        password="pw" ---> becomse secondary key-values
        port="port"
    """
    with open(cfgFile) as f:
        content = f.readlines()
    cfg = {}
    primary_key = 0 # has primary key been discovered (the string enclosed in bracket in config file)
    for line in content:
        if primary_key:
            if '=' in line:
                kv = line.split('=')
                cfg[primary_key].update({kv[0].strip(' "\n'): kv[1].strip(' "\n')})
            else:    
                primary_key = 0
        if (line[0] == '[' and line[-2] == ']'):
            cfg[line[1:-2]] = {}
            primary_key = line[1:-2]
    return cfg

def get_host_connection_details(host_in):
    """ Returns Dictionary of Credentials to Connect to Database 
        Note, it's assumed configuration is in .my.cnf file in home directory
    """
    cfgFile = "~/.my.cnf"
    cfgFile = os.path.expanduser(cfgFile)
    cfg = load_config_file(cfgFile)
    return cfg[host_in]

def create_engine(host_in,schema):
    host_in = host_in.lower()
    cfg = get_host_connection_details(host_in)
    host = cfg['host']
    un = cfg['user']
    pw = cfg['password']
    port = cfg['port']

    if host_in=='redshift':
        url = 'postgresql://'+un+':'+pw+'@'+host+':'+port+'/'+schema
        engine = sqlengine(url)
    else:
        url = 'mysql+pymysql://'+un+':'+pw+'@'+host+':'+port+'/'+schema
        if host_in == 'client':
            url += '?unix_socket=/var/lib/mysql/mysql.sock'
            engine = sqlengine(url)
            #engine=pymysql.connect(host='localhost',user=un,passwd=pw,db='',
            #    port=3306,unix_socket='/var/lib/mysql/mysql.sock')
        else:
            ssl = {'ssl':{'key': cfg['ssl_key'], 'cert': cfg['ssl_cert'], 'ca': cfg['ssl_ca']}}
            engine = sqlengine(url,connect_args=ssl)

    print ("engine created for "+host)
    return engine

def get_dataframe(sql_str,host_in='client',pkl_fn=False):
    if pkl_fn==False:
        db = create_engine(host_in,'')
        df = pd.read_sql(sql_str,db)
    else:
        df = pd.read_pickle(pkl_fn)
    return df

def save_dataframe(df, pkl_fn):
    directory = os.path.dirname(pkl_fn)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_pickle(pkl_fn)

def read_sql(sql_fn,var_replace):
    """ Loads sql statments in sql_fn and replaces variables based on supplied var_replace dictionary 
        SQL statements are returned as string
    """
    with open(sql_fn,'r') as sql:
        sql_stmts = sql.read()
    for key in var_replace:
        sql_stmts = sql_stmts.replace(key,var_replace[key])
    return sql_stmts

def execute_sql(sql_stmt, host_in='client'):
    """ Runs SQL Statements using SQL Alchemy
    """
    #db = create_engine(host_in,'')
    #sql = sqltext(sql_stmt) 
    #return db.execute(sql)
    with open('temp.sql','w') as sql:
        sql.write(sql_stmt)

    proc=sp.Popen("mysql < temp.sql",stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    out,err = proc.communicate()
    sp.Popen("rm temp.sql",stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    return out.strip(),err.strip()

def checksystempiops():
    #For IOPS
    #IOPS-Threshold:12
    return cpu_times_percent(interval=1).iowait

def checksystemcpustats():
    #For CPU-Load
    #CPU-Threshold:40
    return cpu_times_percent(interval=1).idle

def checkDBLocks():
    #DB-LOCKS:10
    proc=sp.Popen("mysql  -e 'SHOW ENGINE INNODB STATUS \G' | grep 'RECORD LOCKS' | wc -l",shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
    out,err = proc.communicate()
    return out.strip(),err.strip()