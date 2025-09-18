import argparse
import json
import os
import re
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
import networkx as nx

# Globals for mapping types to integer IDs
node_type_dict = {}
edge_type_dict = {}

# Metadata defining the file lists for each dataset
metadata = {
    'trace': {
        'train': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3'],
        'test': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3', 'ta1-trace-e3-official-1.json.4']
    },
    'theia': {
        'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1', 'ta1-theia-e3-official-6r.json.2', 'ta1-theia-e3-official-6r.json.3'],
        'test': ['ta1-theia-e3-official-6r.json.8']
    },
    'cadets': {
        'train': ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official.json.1', 'ta1-cadets-e3-official.json.2', 'ta1-cadets-e3-official-2.json.1'],
        'test': ['ta1-cadets-e3-official-2.json']
    },
}

# Pre-compiled regular expressions for parsing log lines
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')
pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')
pattern_netflow_object_address = re.compile(r'remoteAddress\":\"(.*?)\"')
pattern_netflow_object_port = re.compile(r'remotePort\":(.*?),')
pattern_cmd_line = re.compile(r'\"cmdLine\":\"(.*?)\"')
pattern_path = re.compile(r'predicateObjectPath\":\{\"string\":\"(.*?)\"}')
pattern_path2 = re.compile(r'predicateObject2Path\":\{\"string\":\"(.*?)\"}')


def resolve_identifier_name(uuid, pattern_result, id_nodename_map):
    """Helper function to get the best available name for a given UUID."""
    if len(pattern_result) > 0 and pattern_result[0] != '<unknown>':
        return pattern_result[0]
    elif id_nodename_map.get(uuid) is not None and id_nodename_map.get(uuid) != '<unknown>':
        return id_nodename_map.get(uuid)
    else:
        return ""

def construct_provenance_graph(dataset, malicious_uuids, path, is_test=False):
    """Reads a single pre-processed text file and constructs a NetworkX graph."""
    global node_type_dict, edge_type_dict

    # --- Phase 1: Data Aggregation ---
    node_data = defaultdict(lambda: {'semantics': None})
    edge_data = defaultdict(lambda: {'types': set()})

    file_path = f'./{dataset}/{path}.txt'
    print(f'Aggregating data from {file_path} ...')

    with open(file_path, 'r') as f:
        for line in f:
            fields = line.replace('\n', '').split('\t')
            if len(fields) != 8:
                continue

            src, src_type, dst, dst_type, edge_type, ts, cmd_line, path_str = fields
            timestamp = int(ts)

            # In training, filter out events involving known malicious entities.
            if not is_test:
                if src in malicious_uuids and src_type != 'MemoryObject': continue
                if dst in malicious_uuids and dst_type != 'MemoryObject': continue

            # Dynamically update type-to-ID mappings.
            if src_type not in node_type_dict: node_type_dict[src_type] = len(node_type_dict)
            if dst_type not in node_type_dict: node_type_dict[dst_type] = len(node_type_dict)
            if edge_type not in edge_type_dict: edge_type_dict[edge_type] = len(edge_type_dict)
            
            is_reversed = 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type
            u, v, u_type, v_type = (dst, src, dst_type, src_type) if is_reversed else (src, dst, src_type, dst_type)
            
            for node_uuid, node_type in [(u, u_type), (v, v_type)]:
                node_data[node_uuid]['type'] = node_type
                
                if node_type == 'SUBJECT_PROCESS':
                    if node_data[node_uuid]['semantics'] is None:
                        node_data[node_uuid]['semantics'] = defaultdict(set)
                    
                    semantic_feature = ''
                    if cmd_line: semantic_feature += f'{cmd_line} '
                    if path_str: semantic_feature += f'{path_str}'
                    
                    if semantic_feature:
                        node_data[node_uuid]['semantics'][timestamp].add(semantic_feature)

                elif 'FILE' in node_type or node_type == 'NetFlowObject':
                    if node_data[node_uuid]['semantics'] is None:
                        node_data[node_uuid]['semantics'] = set()
                    if path_str:
                        node_data[node_uuid]['semantics'].add(path_str)

            edge_data[(u, v)]['types'].add(edge_type)

    # --- Phase 2: Graph Construction ---
    g = nx.DiGraph()
    node_map = {}
    print("Building graph from aggregated data...")

    for node_uuid, data in tqdm(node_data.items()):
        if node_uuid not in node_map:
            node_map[node_uuid] = len(node_map)
        
        node_id = node_map[node_uuid]
        node_type_id = node_type_dict.get(data['type'])
        final_semantic_features = []

        node_type = data['type']
        if node_type == 'SUBJECT_PROCESS' and data['semantics']:
            sorted_semantics = sorted(data['semantics'].items())
            for _, semantic_set in sorted_semantics:
                final_semantic_features.extend(sorted(list(semantic_set)))
        
        elif ('FILE' in node_type or node_type == 'NetFlowObject') and data['semantics']:
            longest_path = max(data['semantics'], key=len, default='')
            if longest_path:
                final_semantic_features = [longest_path]
                
        g.add_node(node_id,
                   type=node_type_id,
                   semantic_features=final_semantic_features,
                   original_uuid=node_uuid)

    num_all_edge_types = len(edge_type_dict)
    for (u, v), data in tqdm(edge_data.items()):
        if u not in node_map or v not in node_map: continue

        u_id, v_id = node_map[u], node_map[v]
        
        edge_encoding = [0] * num_all_edge_types
        for edge_type in data['types']:
            idx = edge_type_dict[edge_type]
            edge_encoding[idx] = 1

        g.add_edge(u_id, v_id, encoding=edge_encoding)
            
    return node_map, g

def parse_raw_logs(dataset):
    """
    Parses raw JSON logs into an intermediate text format.
    This function is idempotent and will skip files that already exist.
    """
    id_nodetype_map = {}
    id_nodename_map = {}
    
    # This path assumes the raw data is in a specific directory structure.
    # Adjust './Darpa/Engagement3/' if your raw data is located elsewhere.
    raw_data_path = f'./Darpa/Engagement3/{dataset}/'
    
    for file in os.listdir(raw_data_path):
        if 'json' in file and not any(s in file for s in ['.txt', 'names', 'types', 'metadata', 'tar.gz']):
            print(f'Reading {file} to build maps...')
            with open(os.path.join(raw_data_path, file), 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    if any(s in line for s in ['Event', 'Host', 'TimeMarker', 'StartMarker', 'UnitDependency', 'EndMarker']):
                        continue
                    
                    uuid_match = pattern_uuid.findall(line)
                    if not uuid_match: continue
                    uuid = uuid_match[0]
                    
                    type_match = pattern_type.findall(line)
                    if type_match:
                        subject_type = type_match[0]
                    else:
                        if 'MemoryObject' in line: subject_type = 'MemoryObject'
                        elif 'NetFlowObject' in line: subject_type = 'NetFlowObject'
                        elif 'UnnamedPipeObject' in line: subject_type = 'UnnamedPipeObject'
                        else: continue

                    if uuid == '00000000-0000-0000-0000-000000000000' or subject_type == 'SUBJECT_UNIT':
                        continue
                    
                    id_nodetype_map[uuid] = subject_type
                    if 'FILE' in subject_type and pattern_file_name.findall(line):
                        id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                    elif subject_type == 'SUBJECT_PROCESS' and pattern_process_name.findall(line):
                        id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                    elif subject_type == 'NetFlowObject' and pattern_netflow_object_address.findall(line) and pattern_netflow_object_port.findall(line):
                        id_nodename_map[uuid] = f"{pattern_netflow_object_address.findall(line)[0]}:{pattern_netflow_object_port.findall(line)[0]}"

    for key in metadata[dataset]:
        for file in metadata[dataset][key]:
            output_path = f'./{dataset}/{file}.txt'
            if os.path.exists(output_path):
                continue
            
            print(f'Processing {file} into {output_path} ...')
            with open(os.path.join(raw_data_path, file), 'r', encoding='utf-8') as f_in, \
                 open(output_path, 'w', encoding='utf-8') as f_out:
                
                for line in tqdm(f_in):
                    if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                        edgeType_match = pattern_type.findall(line)
                        if not edgeType_match: continue
                        edgeType = edgeType_match[0]
                        
                        timestamp_match = pattern_time.findall(line)
                        if not timestamp_match: continue
                        timestamp = timestamp_match[0]
                        
                        srcId_match = pattern_src.findall(line)
                        if not srcId_match or srcId_match[0] not in id_nodetype_map: continue
                        srcId = srcId_match[0]
                        srcType = id_nodetype_map[srcId]
                        
                        cmdLine_match = pattern_cmd_line.findall(line)
                        cmdLine = resolve_identifier_name(srcId, cmdLine_match, id_nodename_map)

                        dstId1_match = pattern_dst1.findall(line)
                        if dstId1_match and dstId1_match[0] != 'null' and dstId1_match[0] in id_nodetype_map:
                            dstId1 = dstId1_match[0]
                            path1_match = pattern_path.findall(line)
                            path1_str = resolve_identifier_name(dstId1, path1_match, id_nodename_map)
                            dstType1 = id_nodetype_map[dstId1]
                            f_out.write(f"{srcId}\t{srcType}\t{dstId1}\t{dstType1}\t{edgeType}\t{timestamp}\t{cmdLine}\t{path1_str}\n")

                        dstId2_match = pattern_dst2.findall(line)
                        if dstId2_match and dstId2_match[0] != 'null' and dstId2_match[0] in id_nodetype_map:
                            dstId2 = dstId2_match[0]
                            path2_match = pattern_path2.findall(line)
                            path2_str = resolve_identifier_name(dstId2, path2_match, id_nodename_map)
                            dstType2 = id_nodetype_map[dstId2]
                            f_out.write(f"{srcId}\t{srcType}\t{dstId2}\t{dstType2}\t{edgeType}\t{timestamp}\t{cmdLine}\t{path2_str}\n")

    if id_nodename_map:
        with open(f'./{dataset}/uuid_to_name_map.json', 'w', encoding='utf-8') as f:
            json.dump(id_nodename_map, f)
    if id_nodetype_map:
        with open(f'./{dataset}/uuid_to_type_map.json', 'w', encoding='utf-8') as f:
            json.dump(id_nodetype_map, f)

def run_processing_pipeline(dataset):
    """Main function to orchestrate the preprocessing and graph creation."""
    
    # Create the output directory for the dataset if it doesn't exist.
    os.makedirs(f'./{dataset}', exist_ok=True)
    
    malicious_entities_path = f'./{dataset}/{dataset}.txt'
    # Check if the malicious entities file exists, otherwise create an empty one.
    if not os.path.exists(malicious_entities_path):
        print(f"Warning: Malicious entities file not found at {malicious_entities_path}. Assuming no malicious entities.")
        malicious_uuids = set()
        open(malicious_entities_path, 'a').close() # Create the file
    else:
        with open(malicious_entities_path, 'r') as f:
            malicious_uuids = {line.strip() for line in f}

    parse_raw_logs(dataset)

    train_graphs = []
    for file in metadata[dataset]['train']:
        _, train_g = construct_provenance_graph(dataset, malicious_uuids, file, is_test=False)
        train_graphs.append(train_g)

    test_graphs = []
    test_node_map = {}
    node_offset = 0
    for file in metadata[dataset]['test']:
        node_map, test_g = construct_provenance_graph(dataset, malicious_uuids, file, is_test=True)
        test_graphs.append(test_g)
        for uuid, node_id in node_map.items():
            if uuid not in test_node_map:
                test_node_map[uuid] = node_id + node_offset
        node_offset += test_g.number_of_nodes()
    
    final_malicious_entities = []
    malicious_names = []
    name_map_path = f'./{dataset}/uuid_to_name_map.json'
    type_map_path = f'./{dataset}/uuid_to_type_map.json'

    if os.path.exists(name_map_path) and os.path.exists(type_map_path):
        with open(name_map_path, 'r', encoding='utf-8') as f: id_nodename_map = json.load(f)
        with open(type_map_path, 'r', encoding='utf-8') as f: id_nodetype_map = json.load(f)
        
        with open(f'./{dataset}/attack_node_identities.txt', 'w', encoding='utf-8') as f:
            for uuid in malicious_uuids:
                node_type = id_nodetype_map.get(uuid)
                if uuid in test_node_map and node_type and node_type not in ['MemoryObject', 'UnnamedPipeObject']:
                    final_malicious_entities.append(test_node_map[uuid])
                    name = id_nodename_map.get(uuid, uuid)
                    malicious_names.append(name)
                    f.write(f'{uuid}\t{name}\n')

    pkl.dump((final_malicious_entities, malicious_names), open(f'./{dataset}/attack_nodes.pkl', 'wb'))
    pkl.dump([nx.node_link_data(g) for g in train_graphs], open(f'./{dataset}/train_set.pkl', 'wb'))
    pkl.dump([nx.node_link_data(g) for g in test_graphs], open(f'./{dataset}/test_set.pkl', 'wb'))

    with open(f'./{dataset}/node_type_map.json', 'w', encoding='utf-8') as f:
        json.dump(node_type_dict, f, indent=4)
    with open(f'./{dataset}/edge_type_map.json', 'w', encoding='utf-8') as f:
        json.dump(edge_type_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Log Parser and Graph Builder')
    parser.add_argument("--dataset", type=str, default="trace", help="Dataset to process: 'trace', 'theia' or 'cadets'")
    args = parser.parse_args()
    
    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError("Dataset not supported. Please choose from 'trace', 'theia', 'cadets'")
        
    run_processing_pipeline(args.dataset)