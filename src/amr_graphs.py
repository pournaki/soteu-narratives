import re
import igraph as ig
from src.helpers import *
from src.constants import *
import pandas as pd
import numpy as np
from collections import Counter

## build propbank to verbatlas dict
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

## load propbank to pos dict
pbid2pos = pd.read_csv("./data/resources/pbid2pos.csv").set_index('pbid')
english_adjectives = np.loadtxt("./data/resources/english-adjectives.txt",dtype=str)

## legacy code
# def check_predicate_status(input_string):
#     """
#     regex that checks if node string ends with hyphen + digit(s) (e.g. work-01)
#     and does not contain 91
#     identifies propbank predicates
#     """
#     regex_pattern = r'^.+-\d+$'
#     if "91" not in input_string:
#         return bool(re.match(regex_pattern, input_string))
#     else:
#         return False

def check_if_node_is_predicate(graph,node):
    ## predicates are those nodes that have at least 
    ## one outgoing ARG connection
    arg_relations = [":ARG0",":ARG1",":ARG2",":ARG3",":ARG4"]
    neighbors = graph.neighbors(node,mode='out')
    if len(neighbors) > 0:
        pairs = [(node.index,n) for n in neighbors]
        for p in pairs:
            eid = graph.get_eid(p[0],p[1])
            if graph.es[eid]['role'] in arg_relations:
                return True
        return False
    else:
        return False

def check_if_node_is_argof(graph,node):
    ## these nodes are not real predicates, they are part
    ## of modifiers of a concept node
    neighbors = graph.neighbors(node,mode='in')
    pairs = [(n,node.index) for n in neighbors]
    for p in pairs:
        eid = graph.get_eid(p[0],p[1])
        if "-of" in graph.es[eid]['role']:
            return True
    return False

def get_pos(input_string):
    try:
        pos = pbid2pos.loc[input_string.replace('-','.')]['pos']
    except KeyError:
        if input_string in english_adjectives:
            pos = 'j'
        else:
            pos = None
    return pos

def penman_to_igraph(penman_object,amr_string):
    """Transform penman object to igraph object    
    """
    id2inst = {}
    for i in penman_object.instances():
        id2inst[i.source] = i.target
    
    edgelist = []
    for e in penman_object.edges():
        edgelist.append([e.source,e.target,e.role])        
    
    attributes = penman_object.attributes()
    G = ig.Graph.TupleList(edgelist,directed=True,edge_attrs=['role'])
    
    for v in G.vs:
        v['label'] = id2inst[v['name']]
        ## if check_predicate_status(v['label']) == True:
        if check_if_node_is_predicate(G,v) == True:            
            ## make sure it is not part of an arg-of relation
            is_argof = check_if_node_is_argof(G,v)
            if is_argof == False:
                v['type'] = 'predicate'
                try:
                    v['label_va'] = vainfo.loc[pb2va[v['label'].replace("-",".")]][1]
                except KeyError:
                    v['label_va'] = v['label']
            else:
                v['type'] = 'predicate-of'
                try:
                    v['label_va'] = vainfo.loc[pb2va[v['label'].replace("-",".")]][1]
                except KeyError:
                    v['label_va'] = v['label']
        else:
            v['label_va'] = v['label']
            v['type'] = 'concept'

    ## get polarities
    id2negation = {}
    for attr in attributes:
        if attr.role == ":polarity" and attr.target == "-":
            id2negation[attr.source] = True

    ## get quantities
    id2quant = {}
    for attr in attributes:
        if attr.role == ":quant":
            # edgelist.append([attr.source,attr.target,attr.role])
            id2quant[attr.source] = attr.target

    ## get other attributes
    attrdict = {}
    for attr in attributes:
        if attr.role not in [":polarity",":quant"]:
            try:
                attrdict[attr.source] += " " + attr.target.replace('"','')
            except KeyError:
                attrdict[attr.source] = attr.target.replace('"','')
    for v in G.vs:
        try:
            v['label'] = attrdict[v['name']]
        except KeyError:
            pass            
    
    for v in G.vs:
        if v['name'] in id2negation.keys():
            v['polarity'] = "-"
        else:
            v['polarity'] = "+"
        if v['name'] in id2quant.keys():
            v['quant'] = id2quant[v['name']]
        v['pos'] = get_pos(v['label'])

    G['amr_string'] = amr_string
    G['nid2token'] = transform_metadata(penman_object.metadata)
    G['nid2label_pb'] = dict(zip(G.vs['name'],G.vs['label']))
    G['nid2label_va'] = dict(zip(G.vs['name'],G.vs['label_va']))
    return G

def custom_dfs(graph,vertex):
    """Get all simple paths from selected vertex to the root,
       removing paths that are sub-paths of other paths.
    """
    dfs = graph.get_all_simple_paths(vertex,mode='in')
    dfs = remove_sublists(dfs)
    return [i[1:] for i in dfs]

def add_parent_predicates_to_graph(igraph_object):
    """Get all the possible parent predicates of predicates.
    """
    G = igraph_object.copy()
    predicate_indices = np.where(np.array(G.vs['type'])=="predicate")[0]
    for pred_idx in predicate_indices:
        dfs = custom_dfs(G,pred_idx)
        parent_predicates = []    
        for path in dfs:
            for element in path:
                if G.vs['type'][element] == "predicate":
                    parent_predicates.append(element)
                    break

        if len(parent_predicates) > 0:
            G.vs[pred_idx]['parent_predicates_idx'] = parent_predicates
            G.vs[pred_idx]['parent_predicates_labels'] = [G.vs['label'][j] for j in parent_predicates]             
        else:
            G.vs[pred_idx]['parent_predicates_idx'] = None
            G.vs[pred_idx]['parent_predicates_labels'] = None
            
    return G

def extract_event_subgraphs(G,graph_index):
    """From a given graph, get all the event subgraphs
    """
    predicate_indices = np.where(np.array(G.vs['type'])=="predicate")[0]    
    event_subgraphs = []
    event_index = 0
    pred_idx2eid = {}
    event_index_firstpass = 0
    for pred_idx in predicate_indices:
        if len(G.neighbors(pred_idx,mode='out')) > 0:
            pred_idx2eid[pred_idx] = str(graph_index)+"_"+str(event_index_firstpass)
            event_index_firstpass += 1
    
    for pred_idx in predicate_indices:

        if len(G.neighbors(pred_idx,mode='out')) > 0:

            paths = G.get_all_simple_paths(pred_idx,mode='out')
            paths = remove_sublists(paths)

            pruned_paths = []
            for path in paths:
                pruned_path = []
                for idx,element in enumerate(path):
                    if idx > 0:
                        if G.vs['type'][element] == "predicate":
                            pruned_path.append(element)
                            break
                        else:
                            pruned_path.append(element)
                pruned_paths.append(pruned_path)
            event_edgelist = []
            event_edgelist_names = []
            for path in pruned_paths:
                for idx,element in enumerate(path):
                    if idx == 0:
                        event_edgelist.append((pred_idx,element))
                        event_edgelist_names.append((G.vs['name'][pred_idx],G.vs['name'][element]))
                    else:
                        event_edgelist.append((path[idx-1],element))
                        event_edgelist_names.append((G.vs['name'][path[idx-1]],G.vs['name'][element]))
                        
            event_edgelist = list(set(event_edgelist))
            event_edgelist_names = list(set(event_edgelist_names))
                        
            involved_nodes = flatten_list_of_tuples_to_set(event_edgelist)

            ## collect the 91 edges
            new_involved_nodes = []
            for iv_node_idx in involved_nodes:
                dns = G.neighbors(iv_node_idx,mode='in')
                for n in dns:
                    if G.vs['label'][n] in ['have-org-role-91','have-rel-role-91']:
                        simple_paths = G.get_all_simple_paths(n,mode='out')
                        simple_paths_set = list(set([x for xs in simple_paths for x in xs]))
                        new_involved_nodes.extend(simple_paths_set)
                        asg = G.subgraph(new_involved_nodes)
                        for e in asg.es:
                            event_edgelist_names.append((asg.vs['name'][e.source],asg.vs['name'][e.target]))

            ## combine the new 
            involved_nodes = list(set(new_involved_nodes).union(set(involved_nodes)))
            involved_nodes_labels = [G.vs['label'][i] for i in involved_nodes]
            involved_nodes_names = [G.vs['name'][i] for i in involved_nodes]
            
            subgraph = G.subgraph(involved_nodes)
            
            to_delete = []
            for idx,edge in enumerate(subgraph.es):
                if (subgraph.vs['name'][edge.source],subgraph.vs['name'][edge.target]) not in event_edgelist_names:
                    to_delete.append(idx)
            subgraph.delete_edges(to_delete)
            
            ## get the parents
            parents_nodeindices = G.vs[pred_idx]['parent_predicates_idx']
            if parents_nodeindices != None:
                parents_eids = [pred_idx2eid[p] for p in parents_nodeindices]
            else:
                parents_eids = None
            
            nature_pb = G.vs['label'][pred_idx]
            nature_va = G.vs['label_va'][pred_idx]

            event_subgraphs.append({
                                            "id":str(graph_index)+"_"+str(event_index),
                                            "graph":subgraph,
                                            "root_id":G.vs['name'][pred_idx],
                                            "root_label_pb":G.vs['label'][pred_idx],
                                            "root_label_va":nature_va,
                                            "root_polarity":G.vs['polarity'][pred_idx],
                                            "node_ids":involved_nodes_names,
                                            "parents_eids":parents_eids,
                                            })
                        
            event_index += 1
    return event_subgraphs

def get_node_roles(igraph_object,verbose=False):
    """ Get the role of each node in the graph and add it as attribute 'nid2role'
    """
    error = False
    G = igraph_object.copy()
    node_role_dict = {}
    for v in G.vs:

        role = None
        role_inf = None
        
        if verbose == True:
            print(G.vs['label'][v.index])
                    
        direct_neighbors = G.neighbors(v.index,mode='in')               
        nodename = G.vs['label'][v.index]        
        nodeid = G.vs['name'][v.index]
        
        if len(direct_neighbors) == 0:
            role = None
        
        elif len(direct_neighbors) == 1:
            dn = direct_neighbors[0]
            if "91" not in G.vs['label'][dn]:
                eid = G.get_eid(dn,v.index,directed=True)
                role = G.es[eid]['role']
                ## if you have some logical connector like "and"
                ## take over his role
                if "op" in role:
                    neighbors = G.neighbors(dn,mode='in')
                    if len(neighbors)==1:
                        dn_dn = neighbors[0]
                        eid = G.get_eid(dn_dn,dn,directed=True)
                        role_inf = G.es[eid]['role']
            else:
                role = None

        elif len(direct_neighbors) > 1:
        ### TODO: add the case where you have op1's here
            roles = []
            for dn in direct_neighbors:
                if "91" not in G.vs['label'][dn]:
                    for e in G.es:
                        if e.source == dn and e.target == v.index:
                            roles.append(e['role'])
            role = "|".join(roles)            

        ## we might not have to had inferred any roles
        if role_inf == None:
            role_inf = role

        if nodename in logical_connectors:
            role_inf = None
            
        node_role_dict[nodeid] = {'role':role,'role_inf':role_inf}
        
    G['nid2role'] = node_role_dict
    
    return G

def add_predicate_graph(graph):
    """ A helpful representation for complicated AMR graph. 
    Subgraph spanned by predicates and their links.
    """    
    full_hierarchy = {}
    G = graph.copy()
    GC = graph.copy()
    predicates = np.where(np.array(GC.vs['type']) == "predicate")[0]
    edges_to_add = []
    for pred_idx in predicates:
        dfs = custom_dfs(GC,pred_idx)
        parent_predicates = []    
        for path in dfs:
            first_neighbor = path[0]
            if GC.vs['type'][first_neighbor] == "predicate":
                pass
            else:
                path_types = [GC.vs['type'][i] for i in path]
                if "predicate" in path_types:                    
                    for element_idx,element in enumerate(path):         
                        if path_types[element_idx] == "predicate":
                            parent_predicates.append(element)
                            break
                    ## add a new edge between that target node                
                    eid = GC.get_eid(element,path[element_idx-1])
                    role = GC.es[eid]['role']
                    edges_to_add.append([element,pred_idx,role])   
                else:
                    pass
    ## add those edges
    for e in edges_to_add:
        GC.add_edge(e[0],e[1],role=e[2])
    predicate_graph = GC.subgraph(predicates)
    pg_components = predicate_graph.components(mode='weak').subgraphs()
    for c in pg_components:
        degs = np.array(c.degree(mode='in'))
        if 0 in degs:
            root = np.where(degs==0)[0][0]
            distances = c.distances(source=[root])[0]
            distances = [0 if i==np.inf else i for i in distances]
            hierarchy = dict(zip(c.vs['name'],distances))
        else:
            # print("No hierarchy in this subgraph")
            hierarchy = dict(zip(c.vs['name'],np.zeros(len(c.vs))))
        full_hierarchy.update(hierarchy)
    G['hierarchy'] = full_hierarchy    
    G['predicate_graph'] = predicate_graph
    try:
        G['levels'] = max(full_hierarchy.values()) + 1
    except ValueError:
        G['levels'] = 0    
    return G

def add_event_subgraphs(graph,graph_idx):
    """ TODO: Clean up
    """
    full_hierarchy = graph['hierarchy']
    predicate_graph = graph['predicate_graph']
    G = graph.copy()
    nid2polarity = dict(zip(G.vs['name'],G.vs['polarity']))

    if 'predicate' in G.vs['type']:
        
        ## extract the event-subgraphs
        sg_dict = extract_event_subgraphs(G,graph_idx)
        subgraphs_dict = {}
        for sg in sg_dict:
            g = get_node_roles(sg['graph'])
            sg['hierarchy_level'] = full_hierarchy[sg['root_id']]
            sg['graph'] = g
            sg['nid2role'] = g['nid2role']

            nid2actorlabel = {}
            nid2actorgraph = {}
            for k2,v2 in g['nid2role'].items():
                ## TODO: split up actors that have several roles
                if ("ARG" in str(v2['role_inf']) or str(v2['role_inf']) == ":location" ) and "|" not in str(v2['role_inf']) and "-of" not in str(v2['role_inf']):
                    result = node_to_actor_graph(sg['graph'],k2)
                    nid2actorgraph[k2] = result[0]
                    nid2actorlabel[k2] = result[1]

            g['nid2actorlabel'] = nid2actorlabel
            sg['nid2actorlabel'] = nid2actorlabel
            g['nid2actorgraph'] = nid2actorgraph
            sg['nid2actorgraph'] = nid2actorgraph
            
            sg['nid2polarity'] = nid2polarity

            ## get the chain to the root 
            sg_root_id = sg['root_id']
            node = np.where(np.array(predicate_graph.vs['name']) == sg_root_id)[0][0]
            paths = predicate_graph.get_all_simple_paths(node,mode='in')
            paths = remove_sublists(paths)
            paths_labels = []
            for p in paths:
                paths_labels.append([predicate_graph.vs['label_va'][i] for i in p[1:]])
            
            sg['paths_to_root_labels_va'] = paths_labels
            subgraphs_dict[sg['id']] = sg
    
        for sg in sg_dict:
            parents = sg['parents_eids']
            parent_label_va = None
            parent_label_pb = None            
            parent_polarity = None
            if parents != None:
                parent_label_va = []
                parent_label_pb = []                
                parent_polarity = []
                for p in parents:
                    parent_label_va.append(subgraphs_dict[p]['root_label_va'])
                    parent_label_pb.append(subgraphs_dict[p]['root_label_pb'])
                    rid = subgraphs_dict[p]['root_id']
                    parent_polarity.append(subgraphs_dict[p]['nid2polarity'][rid])
                parent_label_va = "|".join(parent_label_va)
                parent_label_pb = "|".join(parent_label_pb)
                parent_polarity = "|".join(parent_polarity)
            else:
                parent_label_pb = None
                parent_label_va = None
            subgraphs_dict[sg['id']]['parent_root_label_pb'] = parent_label_pb                
            subgraphs_dict[sg['id']]['parent_root_label_va'] = parent_label_va
            subgraphs_dict[sg['id']]['parent_root_polarity'] = parent_polarity

        G['subgraphs'] = subgraphs_dict
        G['nid2polarity'] = nid2polarity
    return G

def plot(graph,output=None,margin=50,vertex_label_dist=None,edge_label_dist=None):
    vlabels = ["- " + i['label'] if i['polarity']=="-" else i['label'] for i in graph.vs]
    if output == "None":
        plot_ = ig.plot(graph,
                    # vertex_label=graph.vs['label'], ## change this to G.vs['name'] for the E_INDEX
                    vertex_label=vlabels,
                    vertex_size=30,
                    edge_label=graph.es['role'],
                    edge_label_size=100,
                    edge_label_dist=edge_label_dist,
                    edge_arrow_size=0.6,
                    vertex_label_size=12,
                    vertex_color='white',
                    vertex_frame_width=0,
                    vertex_label_dist=vertex_label_dist,
                    margin=margin,
                    layout=graph.layout_reingold_tilford())
        return plot_
    else:
        plot_ = ig.plot(graph,
                    output,
                    # vertex_label=graph.vs['label'], ## change this to G.vs['name'] for the E_INDEX
                    vertex_label=vlabels,
                    vertex_size=30,
                    edge_label=graph.es['role'],
                    edge_label_size=10,
                    edge_arrow_size=0.6,
                    vertex_label_size=12,
                    vertex_color='white',
                    vertex_label_dist=vertex_label_dist,             
                    edge_label_dist=edge_label_dist,       
                    vertex_frame_width=0,
                    margin=margin,
                    layout=graph.layout_reingold_tilford())
        return plot_


def actor_graph_to_label(graph_input):    
    graph = graph_input.copy()
    nodes_to_discard = []
    label_nodes = []
    nidx2name = dict(zip(np.arange(len(graph.vs)),graph.vs['name']))
    final_string = ""       
    ## remove adjectives
    for v in graph.vs:
        if v['pos'] == 'j':
            nodes_to_discard.append(v.index)
        v['string'] = v['label']
    for e in graph.es:
        if e['role'] == ":poss":
            graph.vs[e.target]['string'] = "of " + graph.vs[e.target]['string']        
    nodes_to_print = np.arange(len(graph.vs))
    nodes_to_print = [i for i in nodes_to_print if i not in nodes_to_discard]
    final_string = ":".join([graph.vs['string'][i] for i in nodes_to_print])
    return final_string

## OLD VERSION THAT WAS NOT SUITABLE FOR FINE-GRAINED ANALYSIS
# def actor_graph_to_label(graph):
#     nodes_to_discard = []
#     label_nodes = []
#     nidx2name = dict(zip(np.arange(len(graph.vs)),graph.vs['name']))
#     for v in graph.vs:
#         if v['pos'] == 'j':
#             nodes_to_discard.append(nidx2name[v.index])
#     for e in graph.es:
#         if e['role'] == ":name":
#             nodes_to_discard.append(nidx2name[e.source])
#             label_nodes.append(nidx2name[e.target])
#         elif e['role'] == ":mod":
#             nodes_to_discard.append(nidx2name[e.target])        
#     n2t = graph['nid2token']
#     order = list(n2t.keys())
#     index_map = {value: index for index, value in enumerate(order)}
    
#     nids_to_print = [i for i in graph.vs['name'] if i not in nodes_to_discard]
#     nids_to_print_sorted = sorted(nids_to_print, key=lambda x: index_map[x])
    
#     final_string = []
#     for idx,n in enumerate(nids_to_print_sorted):
#         if n in label_nodes:
#             final_string.append(graph['nid2label_pb'][n])
#         else:
#             final_string.append(n2t[n])
                
#     return " ".join(remove_consecutive_duplicates(final_string))

def node_to_actor_graph(graph,node):
    nid2label = graph['nid2label_pb']
    node_idx = np.where(np.array(graph.vs['name']) == node)[0][0]
    nodes_to_consider = []
    simple_paths = graph.get_all_simple_paths(node_idx,mode='out')
    nodes_to_consider.extend(list(set([x for xs in simple_paths for x in xs])))            
    ## also check if we have some 91 neighbors
    dns = graph.neighbors(node_idx,mode='in')
    for n in dns:
        if "91" in graph.vs['label'][n]:
            simple_paths = graph.get_all_simple_paths(n,mode='out')
            simple_paths = list(set([x for xs in simple_paths for x in xs]))
            nodes_to_consider.extend(simple_paths)
    nodes_to_consider = list(set(nodes_to_consider))
    concept = nid2label[node]
    
    if len(nodes_to_consider) == 0:
        return concept,concept
    else:
        actor_graph = graph.subgraph(nodes_to_consider)
        return actor_graph,actor_graph_to_label(actor_graph)

def graph_to_edgelist(G):
    final_edgelist = []
    edgelist = G.get_edgelist()
    for t_idx,t in enumerate(edgelist):
        final_edgelist.append((G.vs['label'][t[0]],G.vs['label'][t[1]],G.es['role'][t_idx]))    
    return final_edgelist

def compare_graphs(G1,G2):
    e1 = graph_to_edgelist(G1)
    e2 = graph_to_edgelist(G2)
    return compare_edgelists(e1,e2)

def compare_edgelists(e1,e2):
    return Counter(e1) == Counter(e2)

# def group_identical_objects(objects):
#     groups = []
#     for obj in objects:
#         # Check if obj belongs to any existing group
#         found_group = False
#         for group in groups:
#             if any(compare_graphs(obj, item) for item in group):
#                 group.append(obj)
#                 found_group = True
#                 break
#         # If obj doesn't belong to any existing group, create a new group
#         if not found_group:
#             groups.append([obj])
#     return groups
