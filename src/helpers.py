from collections import Counter
import igraph as ig
import numpy as np
import pandas as pd

def transform_metadata(penman_metadata):
    tokens = penman_metadata['tok'].split(" ")
    nodename2tokenstring = {}
    nodename2tokenidx = dict([(j,int(k)) for j,k in [i.split("~") for i in penman_metadata['alignments'].split(" ")]])
    for k,v in nodename2tokenidx.items():
        nodename2tokenstring[k] = tokens[v]
    return nodename2tokenstring

def list_to_string(lst):
    return "->".join([str(i) for i in lst])

def remove_sublists(input_list):
    paths_strings = [list_to_string(i) for i in input_list]
    to_delete = []
    for ps in paths_strings:
        for other_ps in paths_strings:
            if ps != other_ps:
                if ps in other_ps:
                    to_delete.append(ps)
    return [[int(j) for j in i.split("->")] for i in paths_strings if i not in set(to_delete)]                

def flatten_list_of_tuples_to_set(list_of_tuples):
    return list(set(list(sum(list_of_tuples, ()))))

def flatten_list_of_lists_to_set(list_of_lists):
    return list(set([x for xs in list_of_lists for x in xs]))

def is_subset(d1, d2):
    for key, value in d1.items():
        if key not in d2 or value not in d2[key]:
            return False
    return True

def remove_consecutive_duplicates(lst):
    result = []
    prev = None
    for item in lst:
        if item != prev:
            result.append(item)
            prev = item
    return result

### UPDATED HELPERS TO BUILD CORRECT CSV
def subgraph_to_csvrows(sg,sg_id):
    final_rows = []
    row = {}
    nid2polarity = sg['nid2polarity']
    pred_va = sg['root_label_va']
    pred_pb = sg['root_label_pb']
    pred_pol = nid2polarity[sg['root_id']]
    row['predicate_va'] = pred_va
    row['predicate_pb'] = pred_pb
    row['predicate_pol'] = pred_pol
    row['predicate_nid'] = sg['root_id']
    n2a = sg['nid2actorlabel']
    n2ag = sg['nid2actorgraph']        
    graph = sg['graph']
    root_id = sg['root_id']    
    id2idx = dict(zip(graph.vs['name'],range(len(graph.vs))))
    root_idx = id2idx[root_id]
    root_neighbors_idx = graph.neighbors(root_id,mode="out")
    
    ## make sure to assign the "and" edges properly
    and_neighbors = []
    for n in root_neighbors_idx:    
        name = sg['graph'].vs[n]['label']    
        if name == "and":
            new_neighbors = graph.neighbors(n,mode="out")
            and_neighbors.extend(new_neighbors)
    and_neighbors = list(set(and_neighbors))
    root_neighbors_idx.extend(and_neighbors)
    root_neighbors_idx = list(set(root_neighbors_idx))    
    
    root_neighbors = [graph.vs['name'][i] for i in root_neighbors_idx]    
    roledict = sg['nid2role']
                    
    row['hierarchy_level'] = sg['hierarchy_level']
    row['parent_predicate_va'] = sg['parent_root_label_va']
    row['parent_predicate_pb'] = sg['parent_root_label_pb']    
    row['parent_predicate_pol'] = sg['parent_root_polarity']
    row['amr_string'] = sg['graph']['amr_string']
    row['event_id'] = sg_id
    row['sentence_id'] = "_".join(sg_id.split("_")[:-1])

    try:
        row['parent_id'] = "|".join(sg['parents_eids']) ## there can be more than one parent
    except TypeError:
        row['parent_id'] = sg['parents_eids'] ## there can be more than one parent
    
    for rn in root_neighbors:
        role = str(roledict[rn]['role_inf'])
            
        ## ! DEAL WITH DOUBLE ROLES OF SINGLE NODES LATER !
        if "ARG" in role and "|" not in role and "-of" not in role:

            ## check if the role already exists in the row
            if role not in row.keys():

                row[role] = n2a[rn].lower()
                row[f"{role}_pol"] = nid2polarity[rn]
                row[f"{role}_nid"] = rn
                ## check if there are adjectives in that guy
                actorgraph = n2ag[rn]
                if type(actorgraph) == ig.Graph:
                    if "j" in actorgraph.vs['pos']:
                        adjectives = ""
                        adjectives_pol = ""
                        for vertex in actorgraph.vs:
                            if vertex['pos'] == "j":
                                adjectives+=f"{vertex['label'].lower()}|"
                                adjectives_pol += f"{nid2polarity[vertex['name']]}|"
                        row[f"{role}_adj"] = adjectives[:-1]
                        row[f"{role}_adj_pol"] = adjectives_pol[:-1]
                    else:
                        row[f"{role}_adj"] = np.nan
                        row[f"{role}_adj_pol"] = np.nan
                else:
                    row[f"{role}_adj"] = np.nan
                    row[f"{role}_adj_pol"] = np.nan
            else:
                new_row = row.copy()
                new_row[role] = n2a[rn].lower()
                new_row[f"{role}_pol"] = nid2polarity[rn]
                new_row[f"{role}_nid"] = rn
                ## check if there are adjectives in that guy
                actorgraph = n2ag[rn]
                if type(actorgraph) == ig.Graph:
                    if "j" in actorgraph.vs['pos']:
                        adjectives = ""
                        adjectives_pol = ""                        
                        for vertex in actorgraph.vs:
                            if vertex['pos'] == "j":
                                adjectives+=f"{vertex['label'].lower()}|"
                                adjectives_pol += f"{nid2polarity[vertex['name']]}|"                                
                        new_row[f"{role}_adj"] = adjectives[:-1]
                        new_row[f"{role}_adj_pol"] = adjectives_pol[:-1]
                    else:
                        new_row[f"{role}_adj"] = np.nan
                        new_row[f"{role}_adj_pol"] = np.nan         
                else:
                    new_row[f"{role}_adj"] = np.nan
                    new_row[f"{role}_adj_pol"] = np.nan
                final_rows.append(new_row)
            
    final_rows.append(row)

    return final_rows

def rowlist_to_df(rowlist):
    df = pd.DataFrame(rowlist)     
#     df = df.sort_values(by="subgraph_id")
    df.reset_index(inplace=True)
    df['hierarchy_level'] = df['hierarchy_level'].astype(int)
    column_order = ["event_id",
                    "sentence_id",
                    "predicate_pb",
                    "predicate_va",
                    "predicate_pol",
                    "predicate_nid",
                    "hierarchy_level",
                    "parent_predicate_pb",
                    "parent_predicate_va",
                    "parent_predicate_pol",
                    "parent_id",
                    ":ARG0",
                    ":ARG0_pol",
                    ":ARG0_adj",
                    ":ARG0_adj_pol",
                    ":ARG0_nid",
                    ":ARG1",
                    ":ARG1_pol",
                    ":ARG1_adj",
                    ":ARG1_adj_pol",
                    ":ARG1_nid",
                    ":ARG2",
                    ":ARG2_pol",
                    ":ARG2_adj",
                    ":ARG2_adj_pol",
                    ":ARG2_nid",
                    ":ARG3",
                    ":ARG3_pol",
                    ":ARG3_adj",
                    ":ARG3_adj_pol",
                    ":ARG3_nid",
                    ":ARG4",
                    ":ARG4_pol",
                    ":ARG4_adj",
                    ":ARG4_adj_pol",
                    ":ARG4_nid",                                                                                    
                    "amr_string",
                   ]
    df = df[column_order]
    return df