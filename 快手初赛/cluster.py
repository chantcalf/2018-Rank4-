# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 09:45:02 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_cluster(register,app,video,act):
    devices = list(register['device_type'].unique())
    print (len(devices))
    df = pd.DataFrame({'device_type':devices})
    gp = register.groupby(['device_type']).size().rename('regnum').reset_index()
    df = df.merge(gp,on=['device_type'],how='left')
    
    app = app.merge(register[['user_id','device_type']],on=['user_id'],how='left')
    video = video.merge(register[['user_id','device_type']],on=['user_id'],how='left')
    act = act.merge(register[['user_id','device_type']],on=['user_id'],how='left')
    
    usr = register[['user_id','device_type']]
    gp = app.groupby(['user_id']).size().rename('app#').reset_index()
    usr = usr.merge(gp,on=['user_id'],how='left')
    
    gp = video.groupby(['user_id']).size().rename('video#').reset_index()
    usr = usr.merge(gp,on=['user_id'],how='left')
    
    gp = act.groupby(['user_id']).size().rename('act#').reset_index()
    usr = usr.merge(gp,on=['user_id'],how='left')
    
    gp = video.groupby(['user_id'])['day'].nunique().rename('video_u').reset_index()
    usr = usr.merge(gp,on=['user_id'],how='left')
    
    gp = act.groupby(['user_id'])['day'].nunique().rename('act_u').reset_index()
    usr = usr.merge(gp,on=['user_id'],how='left')
    
    
    gp = app.groupby(['device_type']).size().rename('appnum').reset_index()
    df = df.merge(gp,on=['device_type'],how='left')
    
    gp = video.groupby(['device_type']).size().rename('videonum').reset_index()
    df = df.merge(gp,on=['device_type'],how='left')
    
    gp = act.groupby(['device_type']).size().rename('actnum').reset_index()
    df = df.merge(gp,on=['device_type'],how='left')
    
    
    
    df['appnum'] = df['appnum']/df['regnum']
    df['videonum'] = df['videonum']/df['regnum']
    df['actnum'] = df['actnum']/df['regnum']
    for i in [1,2,3,4,7,10]:
        gp = usr[usr['app#']>=i].groupby(['device_type']).size().rename('app_'+str(i)+'_count').reset_index()
        df = df.merge(gp,on=['device_type'],how='left')
        df['app_'+str(i)+'_count'] = df['app_'+str(i)+'_count']/df['regnum']
    
    for i in [1,2,3]:
        gp = usr[usr['video_u']>=i].groupby(['device_type']).size().rename('video_'+str(i)+'_count').reset_index()
        df = df.merge(gp,on=['device_type'],how='left')
        df['video_'+str(i)+'_count'] = df['video_'+str(i)+'_count']/df['regnum']
    
    
    for i in [1,2,3,4,7]:
        gp = usr[usr['act_u']>=i].groupby(['device_type']).size().rename('act_'+str(i)+'_count').reset_index()
        df = df.merge(gp,on=['device_type'],how='left')
        df['act_'+str(i)+'_count'] = df['act_'+str(i)+'_count']/df['regnum']
    
    df['regnum'] = df['regnum']/register.shape[0]
    
    out = df[['device_type']]
    del df['device_type']
    df= df.fillna(0)
    df = df/df.max()
    
    
    #print (df.describe())
    print ('start')
    kmeans = KMeans(n_clusters=50, random_state=2018).fit(df.values)
    
    out['kmeans'] = kmeans.labels_
    
    return out

if __name__=='__main__':
    register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
    app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
    video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')
    act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')

    register = register[(register['register_day']!=24) | (register['device_type']!=1)]
    register = register[(register['register_day']!=24) | (register['device_type']!=83)]
    register = register[(register['register_day']!=24) | (register['device_type']!=223)]

    out = get_cluster(register,app,video,act)
    out.to_csv('kmeans.csv',index=False)