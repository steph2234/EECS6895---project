
import json
import numpy as np
import pandas as pd
import fastdtw as fastdtw
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

## Word2Vec event embedding results
file = []
for line in open('embedding json_part-00000-5f166835-f8bb-45c0-8549-083ac77275b1-c000.json', 'r'):
    file.append(json.loads(line))

## import patient events sequence data
with open('patient_event.json', 'r') as myfile:
     patient_event = json.load(myfile)



embed_dict = dict()
for i in range(len(file)):
    cur_word = file[i]['word']
    embed_dict[cur_word] = file[i]['vector']['values']


json_dict = []
for i in patient_event:
    current = {}
    current['id'] = i
    current['vector'] = patient_event[i]
    json_dict.append(current)


## similarity between events formulatory code/can use cosine similarity here
def icd_distance(x,y):
        a = np.asarray(embed_dict[x]).reshape(1,-1)
        b = np.asarray(embed_dict[y]).reshape(1,-1)
        return euclidean(a,b)

## patients id 
patient_id = [i for i in patient_event]


## sample similarity 
print(icd_distance('57410','3782.0'))


## get distance between patients
def get_distance(x,y):
   
    a = patient_event[x]
    a_embed = [embed_dict[str(i).lower()] for i in a]
    b = patient_event[y]
    b_embed = [embed_dict[str(i).lower()] for i in b]
    distance, path = fastdtw(a_embed, b_embed, dist=cosine)
        
    return distance


## sample distance between patients generated
print(get_distance('61619','8091'))


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


## Use Spark to expedite the generation of distance matrix
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PySpark Introduction").getOrCreate()
from pyspark.sql import functions as F
from pyspark.sql.types import *


udf_get_distance = F.udf(get_distance)


DF = spark.read.json('json_dict.json')

DF.show()


## Generate pairwise patients events sequence dataframe
df2 = DF.crossJoin(DF).toDF('id1','vector1','id2','vector2')
df_pairs = df2.filter(df2.id1 < df2.id2)


## Generate distance matrix
distance_matrix = df2.withColumn('distance',udf_get_distance(df2.id1,df2.id2).cast(DoubleType()))

##transform to pandas 
results_df = distance_matrix.select('id1','id2','distance').toPandas()


## save it for later use
results_df.to_csv('distance_matrix.csv')


