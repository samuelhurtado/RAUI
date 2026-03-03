import os
import pandas as pd
import fastavro
import time
from tqdm import tqdm
import ollama


# Function to read avro file and convert to pandas dataframe
def avro_to_dataframe(file_path):
    with open(file_path, 'rb') as f:
        reader = fastavro.reader(f)
        records = [record for record in reader if record["source_code"].lower() in sources_of_interest]
    df = pd.DataFrame(records)
    df = df[fields_keep]
    df['titlebody'] = df.apply(lambda row: f"{row['title']}.\n{row['body']}", axis=1)
    return df

fields_keep  = ['an',
                'publication_date',
                'source_code',
                'title',
                'body']

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

# setup and timer
mymodel    = 'jeffh/intfloat-multilingual-e5-large-instruct:f32'
start_time = time.time()

# Get all files in the folder and filter for .avro files
avro_files = [f for f in os.listdir(path_data) if f.endswith('.avro')]

# Open each file, calculate embeddings, join with "an" field, export to parquet

for file_name in avro_files:
    
    file_path = os.path.join(path_data, file_name)
    file_path2 = os.path.join(path_data, "mle5l", file_name.replace(".avro", ""))+"-embeddings-mle5l.parquet"
    print(file_path)
    df = avro_to_dataframe(file_path)
    
    embeddings_list = []
    
    for index, text in tqdm(df['titlebody'].items(), total=len(df['titlebody']), desc="Processing texts"):
        response = ollama.embeddings(model=mymodel, prompt="passage: "+text)
        embeddings_list.append(response['embedding'])
    
    df2 = df[['an']].copy()
    df2["embeddings"] = embeddings_list

    df2.to_parquet(file_path2)

print('time elapsed: ' + str((time.time() - start_time)/60/60) + ' hours')


