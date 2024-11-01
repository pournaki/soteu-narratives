from transition_amr_parser.parse import AMRParser
import csv
import os
import pandas as pd
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(
                    prog='amr_parser',
                    description='Parse AMR strings from raw text',
                    )
parser.add_argument('inputfile')
parser.add_argument('outputfile')
parser.add_argument('-m','--model')

args = parser.parse_args()
INPUT_PATH = args.inputfile
OUTPUT_PATH = args.outputfile
MODEL_NAME = args.model

## suppress the tensor warning
import warnings
warnings.filterwarnings("ignore")

DOCID_FIELD = "doc_id"
SENTID_FIELD = "sentence_id"
TEXT_FIELD = "sentence"

parser = AMRParser.from_pretrained(MODEL_NAME)

## load sentences
print("Loading data...")
df = pd.read_csv(INPUT_PATH)
df = df[df[TEXT_FIELD].notna()]

doc_ids = list(df[DOCID_FIELD])
sentence_ids = list(df[SENTID_FIELD])
documents = list(df[TEXT_FIELD])

print("Tokenizing...")
documents_tokenized = []
for d in tqdm(documents):
    tokens,positions = parser.tokenize(d)
    documents_tokenized.append(tokens)

print("Encoding...")
annotations, machines = parser.parse_sentences(documents_tokenized,jamr=False,no_isi=True)

print("Saving...")
with open(OUTPUT_PATH,"w",encoding="utf-8") as f:
    writer = csv.DictWriter(f,fieldnames=[DOCID_FIELD,SENTID_FIELD,"amr_string"])
    writer.writeheader()
    for idx,annotation in enumerate(annotations):
        doc_id = doc_ids[idx]
        sentence_id = sentence_ids[idx]
        row = {DOCID_FIELD:doc_id, SENTID_FIELD:sentence_id, "amr_string":annotation}
        writer.writerow(row)    
    # for idx,machine in enumerate(machines):
    #     penman_string = machine.get_amr().to_penman(jamr=False, isi=False)
    #     context_id = context_ids[idx]
    #     row = {DOCID_FIELD:context_id, "amr_string":penman_string}
    #     writer.writerow(row)
