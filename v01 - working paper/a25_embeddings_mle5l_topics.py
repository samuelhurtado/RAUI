import os
import pandas as pd
import fastavro
import time
from tqdm import tqdm
import ollama


topic_names = [ 'geopolitical tensions',
                'international trade, tariffs and commercial fragmentation',
                'export markets, external demand, and growth of the global economy',
                'financial markets',
                'prices of energy and other raw materials',
                'inflation, prices and markups',
                'wages and collective bargaining',
                'fiscal policy and debt sustainability',
                'housing market',
                'confidence of agents and evolution of internal demand']

topic_defs  = [ 'geopolitical tensions, armed conflicts, wars, diplomatic crises, military interventions, revolutions and coups',
                'international trade, tariffs, comercial fragmentation, global value chains, international transport of goods, protectionism, globalization',
                'export markets, external demand, growth of the global economy, exports of goods and services',
                'financial markets, the stock market, exchange rates, interest rates, financing conditions, banks and financial institutions, debt, finance, loans',
                'prices of energy and other raw materials, oil, natural gas, electricity',
                'inflation, prices and markups',
                'wages, compensation of employees, collective bargaining, employment conditions and benefits',
                'fiscal policy, sustainability of the public debt, taxes, tax incentives, pensions, public expenditure, fiscal rules, risk premium of public debt, government budgets',
                'housing market, house prices, rents, accessibility to housing, residential construction',
                'confidence of households and firms (particularly about consumption and investment), evolution of internal demand in the economy']
        
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

# for each topic, calculate embeddings, then export all to parquet

embeddings_list = []

for text in topic_defs:
    
    respuesta = ollama.embeddings(model=mymodel, prompt="prompt: "+text)
    embeddings_list.append(respuesta['embedding'])
    
df = pd.DataFrame()
df["embeddings"] = embeddings_list

df.to_parquet('RAUI_embeddings_mle5l_targets.parquet')

print('time elapsed: ' + str((time.time() - start_time)/60/60) + ' hours')


