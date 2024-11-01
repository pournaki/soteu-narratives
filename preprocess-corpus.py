import pandas as pd
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

import argparse
parser = argparse.ArgumentParser(
                    prog='sentence splitter',
                    description='Splits documents into sentences and saves them as a new dataframe',
                    )
parser.add_argument('inputfile')
parser.add_argument('outputfile')
args = parser.parse_args()
INPUT_PATH = args.inputfile
OUTPUT_PATH = args.outputfile

print(f"Reading {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH)

def list_to_len(listo):
    return np.arange(len(listo))
def split_text_into_sentences(string):
    return sent_tokenize(string)

print(f"Splitting documents into sentences...")
df['sentence'] = df['text'].apply(split_text_into_sentences)
df = df.explode("sentence")
df['sent_idx'] = df.groupby('doc_id').cumcount()
df['sentence_id'] = df['doc_id'].astype(str) + "_" + df['sent_idx'].astype(str)
df = df.drop(columns=['text','sent_idx'])

print(f"Saving into {OUTPUT_PATH}...")
df.to_csv(OUTPUT_PATH,index=False)
