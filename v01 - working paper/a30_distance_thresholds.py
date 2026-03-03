import os
import pandas as pd
import fastavro
import time
from tqdm import tqdm
import numpy as np


# Function that computes cosine similarity
def cosine_similarity(vector_1, vector_2):
    return np.dot( np.array(vector_1).flatten(), np.array(vector_2).flatten() ) / ( np.linalg.norm(vector_1) * np.linalg.norm(vector_2) )
        
# Function to read avro file and convert to pandas dataframe
# this version only keeps articles of a minimum length, from the sources that have been selected
def avro_to_dataframe(file_path):
    with open(file_path, 'rb') as f:
        reader = fastavro.reader(f)
        records = [record for record in reader if record["source_code"].lower() in sources_of_interest]
    df = pd.DataFrame(records)
    df = df[fields_keep]
    df['titlebody'] = df.apply(lambda row: f"{row['title']} \n {row['body']}", axis=1)
    df = df[df['titlebody'].str.len() >= 100]
    return df

fields_keep  = ['an',
                'publication_date',
                'source_code',
                'title',
                'body']

# Lista de valores de "newspaper" que te interesan
sources_of_interest = ["vngdia", "abc", "cindas", "eleco", "mundo", "paisn", "expnsi", "razper"]

##################################
##### Main program ###############
##################################

# Paths: read from txt file
temp = []
with open('a00_path.txt') as f:
    temp = f.readlines()
path_data         = temp[1][:-1]                                                                 # avro input directory
del temp
f.close()
assert os.path.exists(path_data)                                                                 # makes sure folder exists    

# read the database

start_time = time.time()

# read the embeddings of the queries for all topics
dft = pd.read_parquet('RAUI_embeddings_mle5l_targets.parquet', engine='pyarrow')

# Get all files in the folder and filter for .avro files
avro_files = [f for f in os.listdir(path_data) if f.endswith('.avro')]

# Open avro file and embeddings file, combine using "an" field

df = pd.DataFrame()
print('the numbers printed here below are times, in seconds')
    
for file_name in avro_files:
    
    file_path = os.path.join(path_data, file_name)
    file_path2 = os.path.join(path_data, "mle5l", file_name.replace(".avro", ""))+"-embeddings-mle5l.parquet"
    print(file_name)
    t1 = time.time()
    dfa = avro_to_dataframe(file_path)
    print('read and select news ' + str(time.time()-t1))
    t1 = time.time()
    dfe = pd.read_parquet(file_path2, engine='pyarrow')
    
    dfa = dfa.merge(
        dfe[["an", "embeddings"]],
        on="an",
        how="left"
    )
        
    del dfe
    dfa = dfa.drop(columns=["source_code", "title", "body"])
    print('read embeddings and join ' + str(time.time()-t1))
    t1 = time.time()
    
    result = []
    if len(dfa)>0:
        for i, topic in dft.iterrows():
            mycol = f'cs_{i+1}'
            mytopicemb = topic['embeddings']
            dfa[mycol] = dfa['embeddings'].apply(lambda x: cosine_similarity(x, mytopicemb))
        
        dfa = dfa.drop(columns=['embeddings'])
        df = pd.concat([df, dfa], ignore_index=True)
        
    del dfa
    print('cosine similarity ' + str(time.time()-t1))
    print(len(df))

df.to_parquet("RAUI_mle5l_10topics_cs.parquet")

topic_thresholds = [0,0,0,0,0,0,0,0,0,0]
n_keep = 25*52*100 # choose here the threshold you want to use (depends on your database, how narrow your topics are, etc)

for i in range(10):
    col = f'cs_{i+1}'
    topic_thresholds[i] = df[col].nlargest(n_keep).min() # high similarity is good
    
dft['threshold']=topic_thresholds
    
dft.to_parquet("RAUI_embeddings_mle5l_targets_thresholds.parquet")