#!/usr/bin/env python
# coding: utf-8

# In[40]:


import json
import numpy as np
import pandas as pd
import fastdtw as fastdtw
from scipy.spatial.distance import cosine


# In[3]:


file = []
for line in open('embedding json_part-00000-5f166835-f8bb-45c0-8549-083ac77275b1-c000.json', 'r'):
    file.append(json.loads(line))


# In[4]:


with open('patient_event.json', 'r') as myfile:
     patient_event = json.load(myfile)


# In[5]:


embed_dict = dict()
for i in range(len(file)):
    cur_word = file[i]['word']
    embed_dict[cur_word] = file[i]['vector']['values']


# In[6]:


json_dict = []
for i in patient_event:
    current = {}
    current['id'] = i
    current['vector'] = patient_event[i]
    json_dict.append(current)


# In[7]:


with open('json_dict.json', 'w') as outfile:
    json.dump(json_dict, outfile)


# In[8]:


def icd_distance(x,y):
        a = np.asarray(embed_dict[x]).reshape(1,-1)
        b = np.asarray(embed_dict[y]).reshape(1,-1)
        return euclidean(a,b)


# In[9]:


patient_id = [i for i in patient_event]


# In[11]:


icd_distance('57410','3782.0')


# In[25]:


def get_distance(x,y):
   
    a = patient_event[x]
    a_embed = [embed_dict[str(i).lower()] for i in a]
    b = patient_event[y]
    b_embed = [embed_dict[str(i).lower()] for i in b]
    distance, path = fastdtw(a_embed, b_embed, dist=cosine)
        
    return distance


# In[24]:


get_distance('61619','8091')


# In[ ]:


## use loop to calculate distance matrix
distance_matrix = pd.DataFrame([], columns=['a', 'b','distance'])
for i in range(len(patient_id)):
    a = patient_event[patient_id[i]]
    a = [str(i).lower() for i in a]
    a = [embed_dict[i] for i in a]
    for j in range(i+1,len(patient_id)):
        b = patient_event[patient_id[j]]
        b = [str(i).lower() for i in b]
        b = [embed_dict[i] for i in b]
        distance_matrix = distance_matrix.append({'a':patient_id[i],'b':patient_id[j],
                                                  'distance':fastdtw(a,b,dist = cosine)[0]},ignore_index = True)


# In[26]:


import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PySpark Introduction").getOrCreate()
from pyspark.sql import functions as F
from pyspark.sql.types import *


# In[27]:


udf_get_distance = F.udf(get_distance)


# In[41]:


DF = spark.read.json('json_dict.json')


# In[42]:


DF.show()


# In[43]:


df2 = DF.crossJoin(DF).toDF('id1','vector1','id2','vector2')


# In[32]:


df_pairs = df2.filter(df2.id1 < df2.id2)


# In[44]:


distance_matrix = df2.withColumn('distance',udf_get_distance(df2.id1,df2.id2).cast(DoubleType()))


# In[ ]:


results_df = distance_matrix.select('id1','id2','distance').toPandas()


# In[ ]:


results_df.to_csv('distance_matrix.csv')


# In[ ]:




