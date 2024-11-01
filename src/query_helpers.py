import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import matplotlib 

args = [":ARG0",":ARG1",":ARG2",":ARG3",":ARG4"]
speakers = ['barroso', 'juncker', 'von_der_leyen']
to_remove = ["this","it","that","some","so","thing"]

renamer = {  "state:have-org-role-91:member": "member state",
             "person:continent:europe": "Europeans",
             "person":"people",
             "government-organization:parliament": "parliament",
             "government-organization:govern-01": "government",
             "i":"I",             
             "person:have-org-role-91:member":"member",
             "government-organization:house":"house"
            }

tid2year = {"0_":2010,
            "1_":2011,
            "2_":2012,
            "3_":2013,
            ##########,
            "4_":2015,
            "5_":2016,
            "6_":2017,
            "7_":2018,
            ##########,            
            "8_":2020,
            "9_":2021,
            "10":2022,
            "11":2023,           
           }

### HELPERS
def get_sentence_id(string):
    return "_".join(string.split("_")[:-1])

def custom_replace(string):
    if string in ["commission","organization:commission"]:
        return "European Commission"
    elif string in ["organization:european union","organization:eu"]:
        return "European Union"
    elif string in ["continent:europe"]:
        return "Europe"
    else:
        return string

def parse_year(string):
    if string[:2] in ["0_","1_","2_","3_"]:
        speaker = "barroso"
    elif string[:2] in ["4_","5_","6_","7_"]:
        speaker = "juncker"
    elif string[:2] in ["11","10","9_","8_"]:
        speaker = "von_der_leyen"
    return speaker    

def parse_speaker(string):
    if string[:2] in ["0_","1_","2_","3_"]:
        speaker = "barroso"
    elif string[:2] in ["4_","5_","6_","7_"]:
        speaker = "juncker"
    elif string[:2] in ["11","10","9_","8_"]:
        speaker = "von_der_leyen"
    return speaker

def custom_split(string):
    try:
        result = string.split("&&")
    except AttributeError:
        result = string
    return result

def add_small_capital(string):
    if string.isupper() == True:
        string = fr"{{\sc {string}}}"
        return string.lower()
    else:
        return string
    
## deal with this later
pb2va = {}
with open ("./data/resources/VerbAtlas-1.1.0/VerbAtlas-1.1.0/pb2va.tsv","r",encoding="utf-8") as f:
    for idx,line in enumerate(f.readlines()):
        if idx > 0:
            pb = line.split(">")[0]
            va = line.split(">")[1].split("\t")[0]
            pb2va[pb] = va

## load verbatlas info
vainfo = pd.read_csv("./data/resources/VerbAtlas-1.1.0/VerbAtlas-1.1.0/VA_frame_info.tsv",skiprows=1,delimiter="\t",header=None)
vainfo = vainfo.set_index(0)            

def pb_to_va(string):
    try:
        return vainfo.loc[pb2va[string.replace("-",".")]][1]
    except KeyError:
        return string    
    
def get_actordf(df):
    argtypes = [':ARG0',':ARG1',':ARG2',':ARG3',':ARG4']
    counts = []
    for argtype in argtypes:
        counts.append(df[argtype].value_counts())
    adf = pd.DataFrame(counts).T  
    adf.columns = argtypes
    adf['total'] = adf.sum(axis=1)
    adf = adf.fillna(0)    
    adf = adf.sort_values(by='total',ascending=False)        
    adf.apply(series_to_int)
    return adf

def series_to_int(series):
    return pd.to_numeric(series,downcast='integer')


def get_argcounts(df):
    sentence_ids = sorted(set(df['sentence_id']))

    arg0_list = []
    arg1_list = []
    pred_list = []

    for sentence_id in sentence_ids:
        sdf = df[df['sentence_id'] == sentence_id]
        preds_pb = list(sdf['predicate_pb'])
        preds_va = list(sdf['predicate_va'])
        pred_list.extend(preds_va)
        arg0s_raw = list(sdf[':ARG0'])
        arg0s_filtered = [i for i in arg0s_raw if type(i) == str and i not in to_remove]
        arg0_list.extend(arg0s_filtered)    
        ## take only the arg1s that are not parent predicates
        arg1s_raw = list(sdf[':ARG1'])
        arg1s_filtered = [i for i in arg1s_raw if i not in preds_pb and type(i)==str and i not in to_remove]    
        arg1_list.extend(arg1s_filtered)
    
    adict = {":ARG0":Counter(arg0_list),
             ":ARG1":Counter(arg1_list),
             ":ARG2":Counter(list(df[df[':ARG2'].notna()][':ARG2'])),
             ":ARG3":Counter(list(df[df[':ARG3'].notna()][':ARG3'])),
             ":ARG4":Counter(list(df[df[':ARG4'].notna()][':ARG4'])),
             }
    
    return adict    

def get_argdf(df,add_predicates=False,N_words=20,latex_style=False):
    sentence_ids = sorted(set(df['sentence_id']))

    arg0_list = []
    arg1_list = []
    pred_list = []

    for sentence_id in sentence_ids:
        sdf = df[df['sentence_id'] == sentence_id]
        preds_pb = list(sdf['predicate_pb'])
        preds_va = list(sdf['predicate_va'])
        pred_list.extend(preds_va)
        arg0s_raw = list(sdf[':ARG0'])
        arg0s_filtered = [i for i in arg0s_raw if type(i) == str and i not in to_remove]
        arg0_list.extend(arg0s_filtered)    
        ## take only the arg1s that are not parent predicates
        arg1s_raw = list(sdf[':ARG1'])
        arg1s_filtered = [i for i in arg1s_raw if i not in preds_pb and type(i)==str and i not in to_remove]    
        arg1_list.extend(arg1s_filtered)
    
    adict = {":ARG0":arg0_list,
             ":ARG1":arg1_list,
             ":ARG2":list(df[df[':ARG2'].notna()][':ARG2']),
             ":ARG3":list(df[df[':ARG3'].notna()][':ARG3']),
             ":ARG4":list(df[df[':ARG4'].notna()][':ARG4']),
             }

    argdf = []
    for arg in args:    
        s = pd.DataFrame.from_dict(Counter(adict[arg]),orient='index').reset_index(names=[arg]).rename(columns={0:arg+'_count'})
        if N_words != None:
            s = s.sort_values(by=arg+'_count',ascending=False)[:N_words].reset_index(drop=True)
        else: 
            s = s.sort_values(by=arg+'_count',ascending=False).reset_index(drop=True)
        argdf.append(s)
    argdf = pd.concat(argdf,axis=1)    
    argdf = argdf.replace(renamer)    

    if latex_style == True:
        for arg in args:
            argdf[arg] = argdf[arg] + "~(" + argdf[arg+"_count"].astype(str) + ")"
    
    if add_predicates == True:
        
        if N_words != None:
            argdf = pd.concat([argdf,pd.DataFrame(df['predicate_va'].value_counts()[:N_words]).reset_index()],axis=1)
        else:
            argdf = pd.concat([argdf,pd.DataFrame(df['predicate_va'].value_counts()).reset_index()],axis=1)            
        argdf = argdf.rename(columns={'predicate_va':'PREDICATE','count':'PREDICATE_count'})
        
        if latex_style == True:
            argdf["PREDICATE"] = argdf['PREDICATE'] + "~(" + argdf["PREDICATE_count"].astype(str) + ")"    
            argdf["PREDICATE"] = argdf['PREDICATE'].apply(add_small_capital)
    
    return argdf

soteu_sentences = pd.read_csv("./data/soteu_speeches_sentences.csv").set_index('sentence_id')
def get_text(sentence_id):
    print(soteu_sentences.loc[sentence_id]['sentence'])
    
def invert_annotation(annot):
    if annot == "A":
        return "B"
    elif annot == "B":
        return "A"    
    else:
        return "N"

annotation = pd.read_csv("./data/resources/annotation/VA_frame_info_annotation_v3.csv")
annotation.set_index("VA_LABEL",inplace=True)    
def annotate_row(row):
    
    inverter = 1
    
    try:
        
        verb_annotation = annotation.loc[row['predicate_va']]['ANNOTATION']
        
        ## invert annotation if we have (double) negations
        if row['predicate_pol'] == "-":
            inverter *= -1
        if type(row['parent_predicate_pol']) == str and row['parent_predicate_pol'] == "-":
            inverter *= -1
        if row[':ARG0_pol'] == "-":
            inverter *= -1
        if row[':ARG1_pol'] == "-":
            inverter *= -1

        if inverter == 1:
            return verb_annotation
        else:
            return invert_annotation(verb_annotation)
    
    except KeyError:
        return "N"
            
#     try:
#         verb_annotation = annotation.loc[row['predicate_va']]['ANNOTATION']

#         if type(row['frame_pol']) == str:

#             if row['predicate_pol'] == "+" and row['frame_pol'] == "+":
#                 return verb_annotation
#             elif row['predicate_pol'] == "-" and row['frame_pol'] == "+":
#                 return invert_annotation(verb_annotation)
#             elif row['predicate_pol'] == "-" and row['frame_pol'] == "-":
#                 return verb_annotation
#             elif row['predicate_pol'] == "+" and row['frame_pol'] == "-":
#                 return invert_annotation(verb_annotation)        
#         else:
#             if row['predicate_pol'] == "+":
#                 return verb_annotation
#             else:
#                 return invert_annotation(verb_annotation)
#     except KeyError:
#         return "N"
    
def soft_aggregation(G):
    edges_to_delete = []
    G_pruned = G.copy()

    G_pruned = G_pruned.components(mode='weak').giant()

    node2starweight = {}

    for v in G_pruned.vs:
        node2starweight[v.index] = 0

    todel = []
    for v in G_pruned.vs:
        neighbors = list(set(G_pruned.neighbors(v)))
        if len(neighbors) == 1:
            todel.append(v.index)
            node2starweight[neighbors[0]] += 1
    G_pruned.delete_vertices(todel)    
    
    od = G_pruned.degree(mode='out')
    new_degree = []
    for i in range(len(G_pruned.vs)):
        new_degree.append(od[i] + node2starweight[i])    
    
    G_pruned.vs['star_degree'] = new_degree
    
    return G_pruned                       


def get_goaldf(df,actor):
    goal_frames = ["OBLIGE_FORCE","REQUIRE_NEED_WANT_HOPE"]    

    direct_goals = df[ (df[':ARG0'].isin(actor)) & (df['predicate_va'].isin(goal_frames)) & (df['predicate_pol'] == "+")]
    framed_goals = df[ (df[':ARG0'].isin(actor)) & (df['parent_predicate_va'].isin(goal_frames)) & (df['parent_predicate_pol'] == "+")]
    dggid = set(direct_goals['sentence_id'])
    framed_goals = framed_goals[~framed_goals['sentence_id'].isin(dggid)]
    direct_goals['goal_type'] = 'direct'
    framed_goals['goal_type'] = 'framed'
    goals = pd.concat([direct_goals,framed_goals])

    wantlist = []
    for idx,row in tqdm(goals.iterrows()):
        
        goal_type = row['goal_type']
        
        ARG0 = row[":ARG0"]
        ARG1 = row[":ARG1"]
        ARG1_nid = row[":ARG1_nid"]
        ARG2_nid = row[":ARG2_nid"]    
        sentence_id = row["sentence_id"]    
        CURRENT_ARG1_nid = ARG1_nid
        CURRENT_ARG2_nid = ARG2_nid    
        PRED = row["predicate_pb"]            
        if goal_type == "direct":
            final_want = [sentence_id,ARG1]
        elif goal_type == "framed":
            final_want = [sentence_id,ARG0,PRED,ARG1]
            
        other_args = []
        
        for arg in [":ARG2",":ARG3",":ARG4"]:
            ## check if there are some args there        
            if type(row[arg]) == str:
                final_want.append(f"A{arg[-1]}:{row[arg]}")
        
        if type(ARG1) == str and len(ARG1)>3 and ARG1[-3] == "-" and not ":" in ARG1:
            
            while True:
                try:                
                    new_row = df[ (df['predicate_nid'] == CURRENT_ARG1_nid) & (df['sentence_id'] == sentence_id)].iloc[0]
                    ARG1_NEXT = new_row[':ARG1']
                    ARG2_NEXT = new_row[':ARG2']
                    CURRENT_ARG1_nid = new_row[':ARG1_nid']
                    CURRENT_ARG2_nid = new_row[':ARG2_nid']
                    final_want.append(ARG1_NEXT)
                    other_args = []
                    for arg in [":ARG2",":ARG3",":ARG4"]:
                        ## check if there are some args there
                        if type(new_row[arg]) == str:
                            final_want.append(f"A{arg[-1]}:{new_row[arg]}")
                except IndexError:
                    break
                try:
                    new_row = df[ (df['predicate_nid'] == CURRENT_ARG2_nid) & (df['sentence_id'] == sentence_id)].iloc[0]
                    ARG2_NEXT = new_row[':ARG2']
                    CURRENT_ARG2_nid = new_row[':ARG2_nid']
                    final_want.append(ARG2_NEXT)
                    other_args = []
                    for arg in [":ARG1",":ARG3",":ARG4"]:
                        ## check if there are some args there
                        if type(new_row[arg]) == str:
                            final_want.append(f"A{arg[-1]}:{new_row[arg]}")
                except IndexError:
                    break
        fw = [i for i in final_want if type(i)==str]
        if len(fw) > 1:
            if fw[1] in actor:
                fw = [fw[0]] + fw[2:]
            wantlist.append(fw)
            
    goaldf = pd.DataFrame(wantlist)
    goaldf['speaker'] = goaldf[0].apply(parse_speaker)        
    goaldf.columns = ['sentence_id','predicate','arg1','argx1','argx2','argx3','argx4','speaker']
    goaldf['predicate_va'] = goaldf['predicate'].fillna("").apply(pb_to_va)
    return goaldf

def compute_tfidf(countmatrix):
    vocab = countmatrix.index
    N_words = len(vocab)    
    mat = np.array(countmatrix).T
    tfidf_matrix = TfidfTransformer().fit_transform(mat)
    tfidf_matrix = tfidf_matrix.toarray()
    dflist = []
    for idx,speaker in enumerate(speakers):
        words = [vocab[i] for i in np.argsort(tfidf_matrix[idx])[::-1][:N_words]]
        scores = np.round(np.sort(tfidf_matrix[idx])[::-1][:N_words],2)    
        column = [words[i] + f" ({scores[i]})" for i in range(N_words)]
        dflist.append(column)
    tfidfdf = pd.DataFrame(dflist).T    
    tfidfdf.columns = ["Emmanuel Barroso","Jean-Claude Juncker","Ursula von der Leyen"]
    return tfidfdf

from matplotlib.colors import LinearSegmentedColormap
# define colors for the colormap
colors = [(220.0/255, 50.0/255, 32.0/255),(128.0/255, 128.0/255, 128.0/255),(0, 90.0/255, 181.0/255)]  # R -> Grey -> G
# define the positions of the colors
positions = [0.0, 0.5, 1.0]
# create the colormap
custom_cmap = LinearSegmentedColormap.from_list("RedGreyBlue", list(zip(positions, colors)))

def compute_edge_score(row):    
    score = (row['B'] - row['A']) / (row['B'] + row['A'])
    return score

def build_graph(df):
    sub = df[ (df[':ARG0'].notna()) & (df[':ARG1'].notna())]
    sub = sub[ ~ ((sub[':ARG0'].isin(to_remove)) | (sub[':ARG1'].isin(to_remove)))]
    sub = sub[[':ARG0',':ARG1','verb_annotation']]
    sub = sub.groupby([':ARG0', ':ARG1', 'verb_annotation']).size().unstack(fill_value=0).reset_index()
    sub = sub[~( (sub['A'] == 0) & (sub['B'] == 0))]
    sub['edge_score'] = sub.apply(lambda row: compute_edge_score(row), axis=1)
    sub['weight'] = sub['A'] + sub['B']
    sub = sub.rename_axis(None, axis=1)
    el = np.array(sub[[':ARG0',':ARG1','edge_score','weight']])
    G = ig.Graph.TupleList(el,edge_attrs=['score','weight'],directed=True)    
    for e in G.es:
        e['color'] = matplotlib.colors.to_rgba(custom_cmap((e['score'] + 1)/2),alpha=0.5)      
    return sub,G    

def get_ego_network(graph,node_name):
    G = graph.copy()
    graph_names = G.vs['name']
    edges_to_delete = []
    for e in G.es:
        s = e.source
        t = e.target
        s_name = graph_names[s]
        t_name = graph_names[t]
        if (s_name != node_name and t_name != node_name) or s == t:
            edges_to_delete.append(e)
    G.delete_edges(edges_to_delete)
    G = G.components(mode='weak').giant()
    for e in G.es:
        e['color'] = matplotlib.colors.to_rgba(custom_cmap((e['score'] + 1)/2),alpha=0.5)  
    return G

def plot_egonet(graph,output=None,bbox=None):
    plot_ = ig.plot(graph,output,
            # vertex_label=graph.vs['label'], ## change this to G.vs['name'] for the E_INDEX
            vertex_label=graph.vs['name'],
#             vertex_size=5*np.sqrt(graph.degree()),
#             vertex_label_size = 5*np.sqrt(graph.degree()),
            edge_color=graph.es['color'],
            edge_width=graph.es['weight'],
            edge_arrow_size=np.sqrt(graph.es['weight']),                    
#             edge_label=graph.es['role'],
#             edge_label_size=100,
#             edge_label_dist=edge_label_dist,
#             edge_arrow_size=0.6,
#             vertex_label_size=12,
            vertex_color='white',
            vertex_frame_width=0,
            vertex_label_color='black',
            bbox=bbox        ,
#             vertex_label_dist=0.5,
            margin=50,
#             layout=graph.layout_reingold_tilford()
           )   
    return plot_        
