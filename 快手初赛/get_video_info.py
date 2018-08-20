# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 16:06:24 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_author_info(act):
    videos = list(act['author_id'].unique())
    print (len(videos))
    
    df = pd.DataFrame({'author_id':videos})
    
    gp = act.groupby(['author_id']).size().rename('act_counts').reset_index()
    df = df.merge(gp,on=['author_id'],how='left')
    
    gp = act.groupby(['author_id'])['user_id'].nunique().rename('user_counts').reset_index()
    df = df.merge(gp,on=['author_id'],how='left')
    
    for i in [0]:
        gp = act[act.action_type==i].groupby(['author_id']).size().rename('action_type'+str(i)+'_counts').reset_index()
        df = df.merge(gp,on=['author_id'],how='left')
    
    
    df = df.sort_values(by='user_counts',ascending = False).reset_index(drop=True)
    df['ranks'] = range(1,len(df)+1)
    print (df[:20])
    print (len(df[df.user_counts>3000]))
    print (len(df[df.user_counts>2000]))

    #out = df[['author_id']]
    
    
    
    
    '''
    
    del df['video_id']
    df= df.fillna(0)    
    kmeans = KMeans(n_clusters=10, random_state=2018).fit(df.values)
    
    out['video_kmeans'] = kmeans.labels_ 
    gp = out.groupby(['video_kmeans']).size()
    print (gp)
    '''
    
    return df
    
    
    
if __name__=='__main__':
    
    act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')


    out = get_author_info(act)
    out.to_csv('author_info.csv',index=False)