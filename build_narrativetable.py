import pandas as pd
import penman
from penman.models import noop
from src.amr_graphs import *
from src.helpers import *
from tqdm import tqdm
import numpy as np

# INPUT_FILE = "./data/soteu_speeches_sentences_amr.csv"
# OUTPUT_FILE = "./results/soteu_narrativetable.csv"

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

## read the amr-parsed corpus
print("Loading AMR-parsed corpus...")
amr_df = pd.read_csv(INPUT_PATH)
amr_df = amr_df[amr_df['amr_string'] != "<NAN>"]

amr_strings = list(amr_df['amr_string'])
text_ids = list(amr_df['sentence_id'])

print("Converting to graphs...")
graphs = {}
subgraphs = {}
for text_id, amr_string in tqdm(zip(text_ids, amr_strings),total=len(text_ids)):
    text_id = '_'.join(text_id.split('-')) #reformat text id

    ## decode the amr-string
    penman_object = penman.decode(amr_string,model=noop.model)
    
    edges = penman_object.edges()
    
    if len(edges) > 0:
        ## transform to graph and process it
        G = penman_to_igraph(penman_object,amr_string)

        ## add parents, predicate graph and event subgraphs
        G = add_parent_predicates_to_graph(G)
        G = add_predicate_graph(G)
        G = add_event_subgraphs(G, text_id)
                
        if 'subgraphs' in G.attributes():        
            graphs[text_id] = G
            subgraphs.update(G['subgraphs'])

print("Building narrative trace table...")
## parse each subgraph into a csv row
df_list = []
for sg_id,sg in subgraphs.items():
    rows = subgraph_to_csvrows(sg,sg_id)
    df_list.extend(rows)
df = rowlist_to_df(df_list)

print("Saving...")
df.to_csv(OUTPUT_PATH,index=False)
