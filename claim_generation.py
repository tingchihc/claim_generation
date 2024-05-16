import argparse
import os
import spacy
import json
import penman
import nltk
import re
import subprocess

nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')

def build_up_bash_file(file_path, outputdir, test_file):
    
    
    output = 'OutputDir=' + outputdir + '\n'
    test = '--test_file ' + current_directory + '/' + test_file
    full_path = current_directory + '/AMRBART/'
    exc_cmd = 'python -u ' + current_directory + '/AMRBART/fine-tune/main.py \\'

    with open(file_path, 'w') as file:
        file.write('export CUDA_VISIBLE_DEVICES=0\n')
        file.write('source activate AMRBART\n')
        file.write('RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"\n')
        file.write('Dataset=examples\n')
        file.write(f'BasePath={full_path}\n')
        file.write('DataPath=$RootDir/../$Dataset\n')
        file.write('ModelCate=AMRBART-large\n')
        file.write('MODEL=$1\n')
        file.write('ModelCache=$BasePath/.cache\n')
        file.write('DataCache=$DataPath/.cache/dump-amr2text\n')
        file.write('lr=2e-6\n')
        file.write('export HF_DATASETS_CACHE=$DataCache\n')
        file.write('if [ ! -d ${DataCache} ];then\n')
        file.write('  mkdir -p ${DataCache}\n')
        file.write('fi\n')
        file.write(output)
        file.write('if [ ! -d ${OutputDir} ];then\n')
        file.write('  mkdir -p ${OutputDir}\n')
        file.write('fi\n')
        file.write('\n')
        file.write(f'{exc_cmd}')
        file.write('\n')
        file.write('    --data_dir $DataPath \\')
        file.write('\n')
        file.write('    --task "amr2text" \\')
        file.write('\n')
        file.write('    ' + test + ' \\')
        file.write('\n')
        file.write('    --output_dir $OutputDir \\')
        file.write('\n')
        file.write('    --cache_dir $ModelCache \\')
        file.write('\n')
        file.write('    --data_cache_dir $DataCache \\')
        file.write('\n')
        file.write('    --overwrite_cache True \\')
        file.write('\n')
        file.write('    --model_name_or_path $MODEL \\')
        file.write('\n')
        file.write('    --overwrite_output_dir \\')
        file.write('\n')
        file.write('    --unified_input True \\')
        file.write('\n')
        file.write('    --per_device_eval_batch_size 8 \\')
        file.write('\n')
        file.write('    --max_source_length 1024 \\')
        file.write('\n')
        file.write('    --max_target_length 400 \\')
        file.write('\n')
        file.write('    --val_max_target_length 400 \\')
        file.write('\n')
        file.write('    --generation_max_length 400 \\')
        file.write('\n')
        file.write('    --generation_num_beams 5 \\')
        file.write('\n')
        file.write('    --predict_with_generate \\')
        file.write('\n')
        file.write('    --smart_init False \\')
        file.write('\n')
        file.write('    --use_fast_tokenizer False \\')
        file.write('\n')
        file.write('    --logging_dir $OutputDir/logs \\')
        file.write('\n')
        file.write('    --seed 42 \\')
        file.write('\n')
        file.write('    --fp16 \\')
        file.write('\n')
        file.write('    --fp16_backend "auto" \\')
        file.write('\n')
        file.write('    --dataloader_num_workers 8 \\')
        file.write('\n')
        file.write('    --eval_dataloader_num_workers 2 \\')
        file.write('\n')
        file.write('    --include_inputs_for_metrics \\')
        file.write('\n')
        file.write('    --do_predict \\')
        file.write('\n')
        file.write('    --ddp_find_unused_parameters False \\')
        file.write('\n')
        file.write('    --report_to "tensorboard" \\')
        file.write('\n')
        file.write('    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run.log ')
        file.write('\n')

def go_down(Nodes, Edges, start_node, start_edge):
    
    totall_ans = []
    check = True
    
    for se in start_edge:
        ans = [] # put edges, nodes in here
        joined_nodes = [] # put node id in here
        joined_edges = [] # put (start_edge, end_edge) in here
        
        ans.append(start_node)
        ans.append(se)
        joined_nodes.append(start_node['id'])
        joined_nodes.append(se['target'])
        joined_edges.append((se['source'], se['target']))
        
        for e in Edges:
            if e['source'] in [sub[1] for sub in joined_edges] and e['edge_info'] != ':coref':
                joined_edges.append((e['source'], e['target']))
                ans.append(e)
                
        for s, e in joined_edges:
            if s not in joined_nodes:
                joined_nodes.append(s)
                
            elif e not in joined_nodes:
                joined_nodes.append(e)
        
        for jn in joined_nodes:
            for n in Nodes:
                if n['id'] == jn and n not in ans:
                    ans.append(n)
        
        
        totall_ans.append(ans)
        
        
        if len(ans) != len(joined_nodes) + len(joined_edges):
            check = False
    
    if check == False:
        return []
    
    else:
        return totall_ans

def transer_to_amr_graph(info, count):
    
    #FILENAME = 'data/single/' + str(count) + '.txt'
    if os.path.exists('tmp/amr/') == False:
        os.mkdir('tmp/amr/')
    FILENAME = 'tmp/amr/' + str(count) + '.txt'

    initial_nodes = []
    initial_edges = []
    
    for i in info:
        if 'id' in i.keys():
            initial_nodes.append(i)
        else:
            initial_edges.append(i)
    
    try:
        data = []
        for i in initial_nodes:
            tmp = (i['id'], ':instance', i['amr_token'])
            data.append(tmp)
    
        for i in initial_edges:
            tmp = (i['source'], i['edge_info'], i['target'])
            data.append(tmp)
    
        graph = penman.Graph(triples=data)
        txt = penman.encode(graph)
        
        # save file
        with open(FILENAME, 'w') as f:
            f.write('# ::id ' + str(count) + '\n')
            f.write('# ::annotator NULL \n')
            f.write('# ::snt NULL \n')
            f.write(txt)
            f.write('\n')
            
        #print('Successfully,', FILENAME)
    except:
        return 0

def collect_amr_graph(Table):
    
    counting = 0
    for start_node, edge_info in Table['traverse'].items():
        ARG = []
        op = []
        others = []
        
        for edge, info in edge_info.items():
            edge_represent = edge.split('_')[0]
            if 'ARG' in edge_represent:
                ARG.append(info)
            
            elif 'op' in edge_represent:
                op.append(info)
            
            else:
                others.append(info)
                
        #print('start_node {}, ARG {}, op {}, others {}'.format(start_node, len(ARG), len(op), len(others)))
        if len(others) == 0:
            data = []

            for arg in ARG:
                for a in arg: 
                    if a not in data:
                        data.append(a)

            for op_list in op:
                for o in op_list:
                    if o not in data:
                        data.append(o)

            # save data as file
            transer_to_amr_graph(data,start_node)
            counting += 1

        else:
            for other in others:
                data = []

                for arg in ARG:
                    for a in arg: 
                        if a not in data:
                            data.append(a)

                for op_list in op:
                    for o in op_list:
                        if o not in data:
                            data.append(o)

                for other_info in other:
                    if other_info not in data:
                        data.append(other_info)

                # save data as file
                transer_to_amr_graph(data,start_node)
                counting += 1
    #print(counting)

def traverse_predicated_nodes(Nodes, Edges, predicated_node):
    
    Table = {}
    error = []
    coref_edge_list = []
    coref_start_nodes = {}
    # ex:
    # predicate_node: edge_info, target: [the edges and nodes under this edge_info]
    # 52-0-24-z5: {ARG1_52-0-24-z6: [], domain_52-0-24-z14: []}
    
    for p in predicated_node:
        Table[p['id']] = {}
    
    for t in Table.keys():
        
        for n in Nodes:
            if n['id'] == t:
                start_node = n
        
        start_edges = []
        coref_edges = []
        for e in Edges:
            if e['source'] == start_node['id'] and 'snt' not in e['edge_info'] and e not in start_edges:
                if e['edge_info'] != ':coref':
                    start_edges.append(e)
                else:
                    if start_node['id'] not in coref_start_nodes.keys():
                        coref_start_nodes[start_node['id']] = 0
                    coref_edges.append(e)
        
        #print('Start_node: ', start_node)
        #print('Start_edge: ', start_edges)
        if coref_edges != []:
            coref_start_nodes[start_node['id']] = len(coref_edges)
            coref_edge_list.append(coref_edges)
        #    print('Start_node: ', start_node)
        #    print('coref_edges: ', coref_edges)
        
        records = go_down(Nodes, Edges, start_node, start_edges)
        #coref_go_down(Nodes, Edges, start_node, coref_edges)
        
        if records != []:
            for r in range(0, len(records)):
                Table[t][start_edges[r]['edge_info'] + '_' + start_edges[r]['target']] = records[r]
        
        else:
            error.append(t)
    
    for e in error:
        del Table[e]

    return Table, coref_edge_list, coref_start_nodes

def is_verb(word):
    
    pos_tag = nltk.pos_tag([word])
    if len(pos_tag) > 0 and pos_tag[0][1].startswith('VB'):
        return True
    else:
        return False

def extract_predicated_nodes(Nodes, Edges):
    
    predicated_node = []
    
    for e in Edges:
        if 'ARG' in e['edge_info']:
            source = e['source']
            for n in Nodes:
                if n['id'] == source and n not in predicated_node:
                    if '-' in n['amr_token'] and len(n['amr_token'].split('-')) == 2 and is_verb(n['word_token']) == True:
                        predicated_node.append(n)
    
    return predicated_node

def is_sentence_reasonable(sentence):
    doc = nlp(sentence)

    # Check if the sentence has a subject and a verb
    has_subject = False
    has_verb = False

    for token in doc:
        if 'subj' in token.dep_:
            has_subject = True
        if 'obj' in token.dep_ or 'ROOT' in token.dep_:
            has_verb = True

    return has_subject and has_verb

def adjust_punctuation(sentence):
    # Remove whitespace before and after punctuation marks
    sentence = re.sub(r'\s+([.,;?!])', r'\1', sentence)
    # Add whitespace after punctuation marks if missing
    sentence = re.sub(r'([.,;?!])(?!\s)', r'\1 ', sentence)
    # Remove duplicate punctuation marks
    sentence = re.sub(r'([.,;?!])\1+', r'\1', sentence)
    # Capitalize the first letter of the sentence
    sentence = sentence.capitalize()
    return sentence

def check_duplicate(j_data):
    ans = {}
    counting = 0

    for k, v in j_data.items():
        if v not in ans.values():
            ans[counting] = v
            counting += 1
    
    return ans

def process_data(input_file, output_file):
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for node in data['nodes']:
        if any(char in node['id'] for char in ':/\\"\''):
            node['id'] = node['id'].replace("/", "").replace(":", "").replace("\\", "").replace('"', "").replace("'","")

    for link in data['links']:
        if any(char in link['source'] for char in ':/\\"\''):
            link['source'] = link['source'].replace("/", "").replace(":", "").replace("\\", "").replace('"', "").replace("'","")
        elif any(char in link['target'] for char in ':/\\"\''):
            link['target'] = link['target'].replace("/", "").replace(":", "").replace("\\", "").replace('"', "").replace("'","")

    NODES = data['nodes']
    EDGES = data['links']
    predicated_nodes = extract_predicated_nodes(NODES, EDGES)

    FILENAME_predicated_nodes = 'tmp/predicate_nodes.jsonl'
    with open(FILENAME_predicated_nodes, "w") as f:
        for item in predicated_nodes:
            f.write(f"{item}\n")

    results, coref_edges, coref_start_nodes = traverse_predicated_nodes(NODES, EDGES, predicated_nodes)
    DATA = {}
    DATA['traverse'] = results
    FILENAME_traverse = 'tmp/traverse.json'
    with open(FILENAME_traverse, 'w') as f:
        json.dump(DATA, f, indent=4)
    
    collect_amr_graph(DATA)

    
    amr = os.path.join('tmp/amr')
    txt_files = os.listdir(amr)
    if len(txt_files) != 0:
        if os.path.exists(os.path.join('tmp', 'preprocess')) == False:
            os.mkdir(os.path.join('tmp', 'preprocess'))
                    
        for t in txt_files:
            input_file = amr + '/' + t
            output_prefix = os.path.join('tmp', 'preprocess') + '/' + t.split('.')[0]
            cmd = 'python AMR_Process/read_and_process.py --config AMR_Process/config-default.yaml --input_file {}  --output_prefix {}'.format(input_file, output_prefix)
            os.system(cmd)    

    jsonl = []
    preprocess_info = os.listdir(os.path.join('tmp', 'preprocess'))

    for p in preprocess_info:
        if 'jsonl' in p:
            jsonl.append(p)

    for j in jsonl:
        jj = os.path.join('tmp', 'preprocess', j)
        with open(jj, 'r') as f:
            json_data = json.load(f)

        json_data['sent'] = ''
        with open(jj, 'w') as f:
            json.dump(json_data, f)

    generation = os.path.join('tmp', 'data4generation.jsonl')
    with open(generation, 'w') as f:
        for jl in jsonl:
            with open(os.path.join('tmp', 'preprocess', jl), 'r') as jf:
                json_data = json.load(jf)
            json.dump(json_data, f)
            f.write('\n')
    
    fp = 'tmp/pipeline_generate_text.sh'
    od = 'tmp/output'
    tf = 'tmp/data4generation.jsonl'
    build_up_bash_file(fp, od, tf)

    command = 'bash ' + fp + ' "xfbai/AMRBART-large-v2"'
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


    ALL = []
    with open('tmp/output/generated_predictions.txt', 'r') as w:
        for line in w:
            ALL.append(line)
                
    original = {}
    total_len = []
    filter_claim = {}

    for a in range(0, len(ALL)):
        reasonable = is_sentence_reasonable(ALL[a])
        if reasonable == True:
            original[a] = ALL[a]
            total_len.append(len(ALL[a]))

    ans = 0
    for t in total_len:
        ans += t
    ans = ans/len(total_len)

    for k, v in original.items():
        if len(v) > ans+20:
            filter_claim[k+1] = adjust_punctuation(v)

    FILENAME = 'tmp/extracted_claim.json'
    with open(FILENAME, 'w') as f:
        json.dump(filter_claim, f, indent=2)
    
    F_FILENAME = output_file
    final_f_claim = {}
    counting_f = 0
    for k, v in filter_claim.items():
        final_f_claim[counting_f] = v
        counting_f += 1

    remove = []
    for k, v in final_f_claim.items():
        if len(v) > 500:
            remove.append(k)
    for r in remove:
        del final_f_claim[r]

    if final_f_claim != {}:
        final_f_claim = check_duplicate(final_f_claim)                    
        with open(F_FILENAME, 'w') as f:
            json.dump(final_f_claim, f, indent=2)
        print('[FINISH] ', F_FILENAME)
        

        summary = ''
        summary_filename = 'tmp/summary.txt' 
        with open(F_FILENAME, 'r') as d:
            ddddata = json.load(d)
        
        for k, v in ddddata.items():
            summary += v[:-1]

        with open(summary_filename, 'w') as file:
            file.write(summary)


def main():
    
    parser = argparse.ArgumentParser(description='Process JSON data from input file and write to output file')
    parser.add_argument('--input', help='Input JSON file path', required=True)
    parser.add_argument('--output', help='Output JSON file path', required=True)
    args = parser.parse_args()
    process_data(args.input, args.output)


if __name__ == '__main__':

    current_directory = os.getcwd()
    main()
    # TODO
    """
    Change the roots in your side
    example command line: 
    python claim_generation.py --input test/graph_no_quotes.json --output test/claim.json
    """
