import os
import pandas as pd
import fastavro
import time
from tqdm import tqdm
import re
from openai import AzureOpenAI


#################################
##### Definitions ###############
#################################

# topic setup

mytopic = 8
mycol = 'cs_'+str(mytopic)


# GPT setup

client = AzureOpenAI(
  azure_endpoint = "your endpoint here", 
  api_key = "your key here",  
  api_version = "2024-02-15-preview"
)

mymodel = "gpt-4o-mini"


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

# function to analyze text with the LLM
def analyze_text_with_llm(text):
    text_prompt = (
    f"This is a news article that talks about fiscal policy and debt sustainability. \n"
    f"Give me a numerical assessment of sentiment (0 if it is very negative, 5 if it is neutral, 10 if it is very positive) "
    f"and uncertainty (0 if there is little uncertainty, 10 if there is a lot of uncertainty) "
    f"that reflects the way this news article talks about \n"
    f"fiscal policy, sustainability of the public debt, taxes, tax incentives, pensions, public expenditure, fiscal rules, risk premium of public debt, government budgets, etc."
    f"(if the news does not talk about any of these topics, simply respond None, None). \n"
    f"Give me a simple answer. If the evaluation is that the sentiment value is Vs and the uncertainty value is Vu, "
    f"give the answer in the following format: \n"
    f"Sentiment Vs, Uncertainty Vu. \n"
    f"These are some examples of hypothetical news headlines and what their assessment might be: \n"
    f"Opposition warns: the increase in military spending will be followed by hidden tax increases (Sentiment 4, Uncertainty 6). \n"
    f"Opposition claims council debt has increased 211% and warns of higher taxes (Sentiment 2, Uncertainty 7). \n"
    f"Local council approves a budget of 130 million with no tax increases that prioritizes social spending, education and the local economy (Sentiment 8, Uncertainty 2). \n"
    f"Opposition demands tax reductions for agriculture in response to the uncertainty over USA tariffs (Sentiment 5, Uncertainty 7). \n"
    f"Trump threatens Harvard with removing its tax-exempt status after freezing 2.2 billion for not yielding to his political demands (Sentiment 3, Uncertainty 8). \n"
    f"Government does not rule out further tax cuts: we may have some room for maneuver (Sentiment 7, Uncertainty 6). \n"
    f"Portugal's Prime Minister presents a program with lower taxes and more public housing (Sentiment 8, Uncertainty 2). \n"
    f"The State grants Navarre greater powers to collect newly created taxes (Sentiment 5, Uncertainty 5). \n"
    f"Senate urges Government to present a budget with the abstention of Sumar, PNV and Junts, without certainty that it can be approved (Sentiment 4, Uncertainty 9). \n"
    f"President reiterates his commitment to presenting a budget this year and works with parliamentary groups (Sentiment 7, Uncertainty 6). \n"
    f"Thousands of people demonstrate in the three Basque capitals for decent pensions (Sentiment 6, Uncertainty 7). \n"
    f"The pension reform passes first review, with warnings about sustainability (Sentiment 7, Uncertainty 8). \n"
    f"Opinion: The improvement in the risk premium is here to stay (Sentiment 7, Uncertainty 5). \n"
    f"The Government begins negotiations on the General State Budget (Sentiment 6, Uncertainty 5). \n"
    f" \n \n This is the press news article to be evaluated: \n \n "
    f"{text} "
    )
    
    response = client.chat.completions.create(
      model=mymodel,
      messages = [{"role":"system", "content":"You are a useful assistant."},
                  {"role":"user","content":text_prompt},])

    result = response.choices[0].message.content
    return result

# Function to extract the values from the answer of the LLM
def extract_values(text):
    text = text.lower()             # lower case
    text = text.replace('\n', ' ')  # substitute line breaks
    text = re.sub(' +', ' ', text)  # get rid of double spaces

    pattern = r'.*?sentiment[\s:\-]*([\d.]+)[^\d]*uncertainty[\s:\-]*([\d.]+)'
    
    matches = re.findall(pattern, text)
    if matches:
        # If multiple matches, we take the last occurrence
        return float(matches[-1][0]), float(matches[-1][1])
    else:
        return None, None
        


##################################
##### Main program ###############
##################################

# Read path from txt file
temp = []
with open('a00_path.txt') as f:
    temp = f.readlines()
path_data    = temp[1][:-1]           # avro input folder
del temp
f.close()

assert os.path.exists(path_data)      # makes sure folder exists    


###############################################################################
#  read data and select rows
###############################################################################

start_time = time.time()

# read the distances and thresholds
dft = pd.read_parquet('RAUI_embeddings_mle5l_targets_thresholds.parquet', engine='pyarrow')
dfd = pd.read_parquet('RAUI_mle5l_10topics_cs.parquet', engine='pyarrow')
dfd = dfd[['an', mycol]]

mythreshold = dft['threshold'][mytopic-1] # has to be 0 for topic 1
myname = 'RAUI_mle5l_topic_'+str(mytopic)

# Get all files in the folder and filter for .avro files
avro_files = [f for f in os.listdir(path_data) if f.endswith('.avro')]

# Open avro file and embeddings file, combine using "an" field

dff = pd.DataFrame()

for file_name in avro_files:
    
    file_path = os.path.join(path_data, file_name)
    print(file_name)
    t1 = time.time()
    dfn = avro_to_dataframe(file_path)
    
    # merge: keep all columns from dfn and dfd, for the rows that have a common value of "an"
    dfx = pd.merge(dfn, dfd, on='an', how='inner')
    
    # apply semantic search
    dfxx = dfx[dfx[mycol] > mythreshold]

    if not dfxx.empty:
        dff = pd.concat([dff, dfxx], ignore_index=True)

    print(len(dff))

del dfn
del dft
del dfx
del dfxx

###############################################################################
#  get sentiment and uncertainty incicators for each row
###############################################################################

# small sample, to check that everything works before launching the full version
# dff = dff.sample(n=200, random_state=4321)

answers = []
sentiments = []
uncertainties = []

for index, text in tqdm(dff['titlebody'].items(), total=len(dff['titlebody']), desc="Processing texts"):
    try:
        answer_ = analyze_text_with_llm(text)
        answers.append(answer_)
        sentiment, uncertainty = extract_values(answer_)
        sentiments.append(sentiment)
        uncertainties.append(uncertainty)
    except Exception as e:
        print(f"Error processing text at index {index}: {e}")
        sentiments.append(None)
        uncertainties.append(None)

dff['answers'] = answers
dff['sentiment'] = sentiments
dff['uncertainty'] = uncertainties

dff.to_parquet(myname + "_gpt_full.parquet")

###############################################################################
#  save a random sample to check it later by hand
###############################################################################

dfs = dff.sample(n=100, random_state=432)
dfs.to_excel(myname + '_gpt_sample100.xlsx')

###############################################################################
#  calculate monthly data
###############################################################################

df = dff[['publication_date', 'sentiment','uncertainty']]

# Just because it had a weird format in our database: convert 'publication_date' to datetime format
df['publication_date'] = pd.to_datetime(df['publication_date'] / 86400000, origin='1970-01-01', unit='D')
df = df.dropna(subset=['publication_date'])

# Percentage of rows that don't have a value for 'sentiment' and 'uncertainty'
pct_rows_no_value_s = 100-(df['sentiment'].notna().sum() / len(df)) * 100
pct_rows_no_value_u = 100-(df['uncertainty'].notna().sum() / len(df)) * 100

print(f"The percentage of rows that don't have a value for sentiment is: {pct_rows_no_value_s:.2f}%")
print(f"The percentage of rows that don't have a value for uncertainty is: {pct_rows_no_value_u:.2f}%")

# keep only the rows with value for sentiment and uncertainty
df = df.dropna(subset=['sentiment'])
df = df.dropna(subset=['uncertainty'])

# Create year_month column
df['year_month'] = df['publication_date'].dt.strftime('%Y-%m')

# Count monthly articles and calculate monthly means
monthly_count = df.groupby('year_month').size().reset_index(name='item_count')
monthly_means = df.groupby('year_month')[['sentiment', 'uncertainty']].mean().reset_index()

# Calculate the percentage of rows with each sentiment value (0 to 10) for each month
sentiment_counts = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
sentiment_percentages = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0)
sentiment_percentages.columns = [f'sentiment_{int(col)}' for col in sentiment_percentages.columns]
df_sentiment_distribution = sentiment_percentages.reset_index()

# Calculate the percentage of rows with each uncertainty value (0 to 10) for each month
uncertainty_counts = df.groupby(['year_month', 'uncertainty']).size().unstack(fill_value=0)
uncertainty_percentages = uncertainty_counts.div(uncertainty_counts.sum(axis=1), axis=0)
uncertainty_percentages.columns = [f'uncertainty_{int(col)}' for col in uncertainty_percentages.columns]
df_uncertainty_distribution = uncertainty_percentages.reset_index()

# Combine monthly_means with monthly_distributions into a single dataframe
combined_df = pd.merge(monthly_count, monthly_means, on='year_month')
combined_df = pd.merge(combined_df, df_sentiment_distribution, on='year_month')
combined_df = pd.merge(combined_df, df_uncertainty_distribution, on='year_month')

# save monthly series in excel file
combined_df.to_excel(myname + "_gpt_montlyseries.xlsx", index=False)

print('time elapsed: ' + str((time.time() - start_time)/60/60) + ' hours')





