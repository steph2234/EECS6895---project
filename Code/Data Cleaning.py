
import pandas as pd
import numpy as np
import datetime
from pandas import Timestamp
from pandas import Timedelta

df = pd.read_csv("/CarePre/ADMISSIONS.csv")


from os import listdir
from os.path import isfile, join
mypath = "/CarePre"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


prescriptions = pd.read_csv("/CarePre/PRESCRIPTIONS.csv",low_memory=False)

patients = pd.read_csv("/CarePre/PATIENTS.csv")

admissions = pd.read_csv('/CarePre/ADMISSIONS.csv')

diagnoses = pd.read_csv('/CarePre/DIAGNOSES_ICD.csv')

procedures = pd.read_csv('/CarePre/PROCEDURES_ICD.csv')

procedures = pd.read_csv('/CarePre/PROCEDURES_ICD.csv')

D_procedures_icd = pd.read_csv('/CarePre/D_ICD_PROCEDURES.csv')

CPT = pd.read_csv('/CarePre/CPTEVENTS.csv',low_memory = False)

D_diag_icd = pd.read_csv('/CarePre/D_ICD_DIAGNOSES.csv')

diag_icd = pd.read_csv('/CarePre/DIAGNOSES_ICD.csv')



merge1 = pd.merge(admissions[['SUBJECT_ID','HADM_ID','ADMITTIME','ADMISSION_TYPE']],diag_icd, how = 'left',on = 'HADM_ID')
merge2 = pd.merge(merge1,D_diag_icd[['ICD9_CODE','SHORT_TITLE']],how = 'left',on = 'ICD9_CODE')

##Dataframe for diagnose events
diag_df = merge2.rename(columns = {'SUBJECT_ID_x':'SUBJECT_ID','ICD9_CODE':'ICD_9'}).drop(['SUBJECT_ID_y'],axis = 1)


merge3 = pd.merge(admissions[['SUBJECT_ID','HADM_ID','ADMITTIME','ADMISSION_TYPE']],procedures,
                    how = 'left',on = 'HADM_ID')
merge4 = pd.merge(merge3,D_procedures_icd[['ICD9_CODE','SHORT_TITLE']],how = 'left',on = 'ICD9_CODE')

##Dataframe for Procedure events
procedure_df = merge4.rename(columns = {'SUBJECT_ID_x':'SUBJECT_ID','ICD9_CODE':'ICD_9'}).drop(['SUBJECT_ID_y'],axis = 1)

##Dataframe for Prescription events
prescriptions_df = prescriptions[['SUBJECT_ID','HADM_ID',
                                  'STARTDATE','FORMULARY_DRUG_CD']].rename(columns = {'STARTDATE':'ADMITTIME'})

## Tranform into json dict for future use
patient_dict = dict()
for index in admissions.SUBJECT_ID.unique():
    patient = diag_df[diag_df['SUBJECT_ID'] == index][['ADMITTIME','ICD_9','SHORT_TITLE']]
    patient_dict[index] = []
    for i in range(len(patient)):
        stay = patient.iloc[i,:].to_dict()
        stay['event_type'] = 'Diagnose'
        patient_dict[index].append(stay)

procedure_dict = dict()
for index in admissions.SUBJECT_ID.unique():
    patient = procedure_df[procedure_df['SUBJECT_ID'] == index][['ADMITTIME','ICD_9','SHORT_TITLE']]
    procedure_dict[index] = []
    for i in range(len(patient)):
        stay = patient.iloc[i,:].to_dict()
        stay['event_type'] = 'Procedure'
        procedure_dict[index].append(stay)

prescriptions_dict = dict()
for index in admissions.SUBJECT_ID.unique():
    patient = prescriptions_df[prescriptions_df['SUBJECT_ID'] == index][['ADMITTIME','FORMULARY_DRUG_CD']]
    prescriptions_dict[index] = []
    for i in range(len(patient)):
        stay = patient.iloc[i,:].to_dict()
        stay['event_type'] = 'Prescriptions'
        prescriptions_dict[index].append(stay)


## All features
result_dict = dict()
for index in admissions.SUBJECT_ID.unique():
    result_dict[index] = patient_dict[index]+procedure_dict[index]+prescriptions_dict[index]


## extract event sequence based on ICD-9 code range as needed
respiratory = list(diag_df[diag_df['ICD_9'].isin([str(x) for x in np.arange(520,580)])]['SUBJECT_ID'])

respira_dict = dict()
for index in respiratory:
    respira_dict[index] = result_dict[index]


## drop unused features
for patient in respira_dict:
    try:
        for event in respira_dict[patient]:
            event.pop('event_type')
            if event.get('SHORT_TITLE'):
                event.pop('SHORT_TITLE')
            if event.get('FORMULARY_DRUG_CD'):
                event['ICD_9'] = event.pop('FORMULARY_DRUG_CD')
    except:pass


##create input list for Word2Vec model; considering time window as 7 days
input_list = []
for patient_id in respira_dict:
    a = pd.DataFrame.from_dict(respira_dict[patient_id])
    a['Time'] = pd.to_datetime(a['ADMITTIME'],format = '%Y-%m-%d %H:%M:%S')
    patient = a.iloc[:,1:3]
    patient = patient.sort_values(by='Time')
    array = patient.values
    for i in range(len(array)):
        event = array[i]
        time = event[1]
        for j in np.arange(i+1,len(array)):
            related_event = array[j]
            compare_time = related_event[1]
            if compare_time - time <= Timedelta('7 days 00:00:00'):
                input_list.append([event[0],related_event[0]])
            else:
                break
