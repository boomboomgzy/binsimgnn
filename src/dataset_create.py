import programl
import os
from enum import Enum
import torch
from torch_geometric.data import HeteroData
from programl.proto import ProgramGraph
from typing import Any, Dict, Iterable, Optional, Union
from programl.util.py.executor import ExecutorLike, execute
from concurrent.futures import ThreadPoolExecutor
from llvmlite import binding
import subprocess
from tqdm import tqdm
import multiprocessing
import sys
import numpy as np
import shutil
from pathlib import Path
import random
from gensim.models import Word2Vec
import re
from sklearn.preprocessing import normalize
from programl.proto import program_graph_pb2
from collections import defaultdict
import itertools

prop_threads=10 #不要开太大  不然内存占用会偏大

model = Word2Vec(vector_size=16, sg=0,window=8, min_count=1, workers=prop_threads) #参数需要做进一步实验看最好的效果


class BatchProgramlCorpus:
    def __init__(self, ir_programl_list):
        self.ir_programl_list=ir_programl_list

    def __iter__(self):
        for ir_programl in self.ir_programl_list:
                for node in ir_programl.node:
                    if node.type==0:
                        if inst_node_isvalid(node):#只收集有效inst节点的token
                            #该节点特殊处理
                            if node.text=='[external]':
                                yield node.text.split() 
                            else:
                                node_text = node.features.feature["full_text"].bytes_list.value[0].decode('utf-8')
                                node_text=normalize_inst(node_text)
                                node.text=node_text  #node.text 修改为归一化后的full_text
                                yield node_text.split()

                    elif node.type==3:
                        pass
                    else:
                        yield [node.text]



def normalize_inst(inst):
    inst=re.sub(r'@global_var_\d+', '@global_var', inst)
    inst=re.sub(r'@\d+', '@global_var', inst)
    inst=re.sub(r'%[\w\.\-]+', '%ID', inst)
    inst=re.sub(r'(?<=\s)-?\d+(\.\d+)?\b', '%CONST', inst)
    return inst

def remove_metadata_comma_align(ir):
    #delete the meta info 
    ir=re.sub(r', ?!insn\.addr !\d+', '', ir)
    ir=re.sub(r', align \d+','',ir)
    return ir

def inst_node_isvalid(inst_node): #该指令是否有用
    #该节点要特殊处理
    if inst_node.text=='[external]':
        return True
    if inst_node.features and inst_node.features.feature["full_text"].bytes_list.value :     
        inst_node_text = inst_node.features.feature["full_text"].bytes_list.value[0].decode('utf-8')
        if inst_node_text=='': #无用节点
            return False
        else:
            return True



def programG2pyg(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> None:


            
    def _run_one(graph: ProgramGraph) -> None:

            # 4 lists, one per edge type
        # (control,input,output,call) control:inst->inst  input:data->inst output:inst->data   call:inst->inst
            edge_index = [[], [], [],[]] 
            edge_positions = [[], [], [],[]]
            # 2 node type: data instruction
            data_nodes=[]
            inst_nodes=[]
            data_node_index_map={}
            inst_node_index_map={}
            unvalid_inst_nodes_global_idx=[]

            function_dict={}
            for function in graph.function:
                function_dict[function.name]={'data':[],'inst':[]}


            for global_idx,node in enumerate(graph.node):
                if node.type==0:
                    if inst_node_isvalid(node):
                        inst_nodes.append(node)
                        inst_idx=len(inst_nodes)-1
                        inst_node_index_map[str(global_idx)]=inst_idx
                        function_dict[graph.function[node.function].name]['inst'].append(inst_idx)
                    else:
                        unvalid_inst_nodes_global_idx.append(global_idx)
                elif node.type==3:
                    pass
                else:
                    data_nodes.append(node)
                    data_idx=len(data_nodes)-1
                    data_node_index_map[str(global_idx)]=data_idx
                    function_dict[graph.function[node.function].name]['data'].append(data_idx)


            # Create the adjacency lists and the positions
            for edge in graph.edge:
                e_type=edge.flow
                source_node=edge.source
                target_node=edge.target
                #过滤和无效inst节点有关的边
                if (source_node in unvalid_inst_nodes_global_idx ) or (target_node in unvalid_inst_nodes_global_idx):
                    continue
                t=-1
                if e_type==0:# inst->inst
                    edge_index[0].append([inst_node_index_map[str(source_node)],inst_node_index_map[str(target_node)]])
                    t=0
                elif e_type==1:
                    if graph.node[source_node].type==0:# output: inst->data
                        edge_index[2].append([inst_node_index_map[str(source_node)],data_node_index_map[str(target_node)]])
                        t=2
                    else:#input: data->inst
                        edge_index[1].append([data_node_index_map[str(source_node)],inst_node_index_map[str(target_node)]])
                        t=1
                elif e_type==2: #inst->inst
                    edge_index[3].append([inst_node_index_map[str(source_node)],inst_node_index_map[str(target_node)]])
                    t=3
                else: 
                    pass
                #边和position一一对应
                edge_positions[t].append(edge.position)


            # Pass from list to tensor

            edge_index = [torch.tensor(ej) for ej in edge_index]
            edge_positions = [torch.tensor(edge_pos_flow_type) for edge_pos_flow_type in edge_positions]


            # Create the graph structure
            hetero_graph = HeteroData()


            # Add the adjacency lists
            hetero_graph['inst', 'control', 'inst'].edge_index = edge_index[0].T.contiguous()
            hetero_graph['data', 'input', 'inst'].edge_index = edge_index[1].T.contiguous()
            hetero_graph['inst', 'output', 'data'].edge_index = edge_index[2].T.contiguous()
            hetero_graph['inst', 'call', 'inst'].edge_index = edge_index[3].T.contiguous()

            # Add the edge positions
            hetero_graph['inst', 'control', 'inst'].edge_attr = edge_positions[0]
            hetero_graph['data', 'input', 'inst'].edge_attr = edge_positions[1]
            hetero_graph['inst', 'output', 'data'].edge_attr = edge_positions[2]
            hetero_graph['inst', 'call', 'inst'].edge_attr = edge_positions[3]
            
            
            hetero_graph.programl_graph=graph #保存这个programl图用于生成节点的特征向量
            hetero_graph.function_dict=function_dict

            #subdir_binname  作为图的label 如果相同则相似 否则不相似
            heteroG_file_path=graph.module[-1].name
            match=re.search(r'O[0-3sfast]+_(.*?)\.strip', heteroG_file_path)
            bin_name=match.group(1)
            subdir=os.path.basename(os.path.dirname(heteroG_file_path))
            hetero_graph.g_label=subdir+'_'+bin_name
            
            #保存heteroG
            heteroG_subdir_path=os.path.dirname(heteroG_file_path)
            os.makedirs(heteroG_subdir_path,exist_ok=True)

            torch.save(hetero_graph,heteroG_file_path)


    if isinstance(graphs, ProgramGraph):
        _run_one(graphs)
        return

    execute(_run_one, graphs, executor, chunksize)



def ir_file_preprocess(ll_file_path,log_dir,save_dir):
    ir_string=''
    with open(ll_file_path,'r') as file:
        ir_string=file.read()
        # 因llvm版本不一致导致的报错 暂且如此处理
        ir_string = ir_string.replace("%wide-string", "zeroinitializer")
        ir_string = remove_metadata_comma_align(ir_string)
        ir_string = "\n".join([line for line in ir_string.splitlines() if "uselistorder" not in line])
        dir_name=os.path.basename(os.path.dirname(ll_file_path))
        ll_file_name=os.path.splitext(os.path.basename(ll_file_path))[0] #无后缀文件名
        save_sub_dir=os.path.join(save_dir,dir_name)
        log_sub_dir=os.path.join(log_dir,dir_name)
        os.makedirs(save_sub_dir, exist_ok=True)
        os.makedirs(log_sub_dir, exist_ok=True)
       
        e=ir_validation(ir_string) 
        if e==None:
            save_file_path=os.path.join(save_sub_dir,ll_file_name+'.preprocessed.ll')
            with open(save_file_path,'w') as f:
                f.write(ir_string)
        else: #记录无效原因
            log_file_path=os.path.join(log_sub_dir,ll_file_name+'.err_log')
            with open(log_file_path,'w') as f:
                f.write(str(e))

def ir_validation(ir_string):
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    try:
        llvm_module = binding.parse_assembly(ir_string)
        llvm_module.verify()
    except Exception as e:
        return e
    
    return None

def read_vector_from_file(file_path):
    with open(file_path, 'r') as file:
        vector_string = file.readline().strip()
        vector = np.array(list(map(float, vector_string.split())))
    return vector

def gen_tokens(ir_programls_list):

    global model
    corpus = BatchProgramlCorpus(ir_programls_list)
    if len(model.wv) == 0:
        model.build_vocab(corpus)
    else:
        model.build_vocab(corpus, update=True)
        
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

#预处理后的IR
def ll2programl(ll_file_dir,ir_programl_dir,heteroG_save_dir):

    for subdir in os.listdir(ll_file_dir): 
        heteroG_file_path_list=[] #保存的heteroG 的pth文件
        sub_dir_path = os.path.join(ll_file_dir, subdir)
        ll_file_path_list=[] #当前subdir下所有ll_file
        heteroG_save_sub_dir_path=os.path.join(heteroG_save_dir,subdir)
        os.makedirs(heteroG_save_sub_dir_path,exist_ok=True)
        for ll_file in os.listdir(sub_dir_path):
            ll_file_path=os.path.join(sub_dir_path,ll_file)
            ll_file_name=os.path.splitext(os.path.basename(ll_file_path))[0]
            ll_file_path_list.append(ll_file_path)
            heteroG_file_path_list.append(os.path.join(heteroG_save_sub_dir_path,ll_file_name+'.pth'))

        ir_programls=None        

        def read_ll_files(ll_file_path_list):
            for ll_file_path in ll_file_path_list:
                with open(ll_file_path, 'r') as f:
                    yield f.read()


        with ThreadPoolExecutor(max_workers=prop_threads) as executor:
            ir_programls=list(programl.from_llvm_ir(read_ll_files(ll_file_path_list), executor=executor,chunksize=prop_threads+1))

        #创建子目录
        prog_save_sub_dir=os.path.join(ir_programl_dir,subdir)
        os.makedirs(prog_save_sub_dir,exist_ok=True)

        for id,ir_prog in enumerate(ir_programls):
            new_module = program_graph_pb2.Module()
            new_module.name = heteroG_file_path_list[id]
            ir_prog.module.append(new_module)
            file_name=os.path.splitext(os.path.basename(heteroG_file_path_list[id]))[0]
            prog_save_path=os.path.join(prog_save_sub_dir,file_name+'.prog')
            programl.save_graphs(prog_save_path,[ir_prog])

        gen_tokens(ir_programls)

#ll_file_dir为预处理过的IR所在目录
def programl2heteroG(ir_programl_dir):

    def load_prog_data(ir_programl_dir):
    
        for subdir in os.listdir(ir_programl_dir):
            sub_dir_path = os.path.join(ir_programl_dir, subdir)
            for prog_file in os.listdir(sub_dir_path):
                prog_data_path=os.path.join(ir_programl_dir, subdir, prog_file)
                yield programl.load_graphs(prog_data_path)[0]



    prog_iter=load_prog_data(ir_programl_dir)


    with ThreadPoolExecutor(max_workers=prop_threads) as executor:
        try:
            executor.map(programG2pyg, prog_iter)
        except Exception as e:
            print(f"Exception occurred during map execution: {e}")






def prep_ir_file(ll_file_dir,prep_save_dir,prep_log_dir):

    for subdir in os.listdir(ll_file_dir): #每个subdir 分开处理 
        sub_dir_path = os.path.join(ll_file_dir, subdir)
        ll_file_path_list=[] #当前subdir下所有ll_file
        if os.path.isdir(sub_dir_path):
            for ll_file in os.listdir(sub_dir_path):
                ll_file_path=os.path.join(sub_dir_path,ll_file)
                if os.path.isfile(ll_file_path):
                    ll_file_path_list.append(ll_file_path)
                else:
                    print('subdir can just have IR file: ',sub_dir_path)
                    sys.exit(1)
        else:
            print('it must be a dir: ',sub_dir_path)
            sys.exit(1)

        #先做预处理 剔除无效的IR
        with ThreadPoolExecutor(max_workers=prop_threads) as executor:
            futures = [executor.submit(ir_file_preprocess, ll_file_path, prep_log_dir,prep_save_dir) for ll_file_path in ll_file_path_list]
            for future in futures:
                try:
                    future.result() 
                except Exception as e:
                    print(f"An error occurred during processing: {e}")


def collect_heteroG_files(root_dir):
    # 遍历所有子目录，收集所有的 .pth 文件路径
    #heteroG_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pth'):
                yield os.path.join(subdir,file)
    #           heteroG_files.append(os.path.join(subdir, file))
    #return heteroG_files

def init_nodevector(heteroG_save_dir):

    heteroG_files=collect_heteroG_files(heteroG_save_dir)
    #为每个异构图初始化节点向量
    for heteroG_file in heteroG_files:
        heteroG=torch.load(heteroG_file)
        programl_graph=heteroG.programl_graph
        
        data_nodes=[]
        inst_nodes=[]
        #根据node.text初始化节点特征向量
        inst_features=[]
        data_features=[]

        for node in programl_graph.node:
            if node.type==0:
                if inst_node_isvalid(node):
                    inst_nodes.append(node)
            elif node.type==3:
                pass
            else:
                data_nodes.append(node)

        for node in inst_nodes:
            inst=node.text
            token_list=inst.split()
            token_vectors = [model.wv[token] for token in token_list]
            #node_vector=np.sum(token_vectors, axis=0)
            node_vector=np.mean(token_vectors,axis=0)
            #可以试试根据词频来加权
            inst_features.append(node_vector)

        for node in data_nodes:
            data=node.text
            node_vector=model.wv[data]
            data_features.append(node_vector)


        heteroG['inst'].x= torch.tensor(np.array(inst_features), dtype=torch.float)
        heteroG['data'].x= torch.tensor(np.array(data_features), dtype=torch.float)
        
        #初始化完后可以删除heteroG.programl_graph
        delattr(heteroG,'programl_graph')
        torch.save(heteroG, heteroG_file)

def build_dataset(heteroG_save_dir,heteroG_dataset_dir,dataset_size):
    
    heteroG_files=collect_heteroG_files(heteroG_save_dir)



    g_label_to_files = defaultdict(list)
    for file in heteroG_files:
        match=re.search(r'O[0-3sfast]+_(.*?)\.strip', file)
        bin_name=match.group(1)
        subdir=os.path.basename(os.path.dirname(file))
        g_label=subdir+'_'+bin_name
        g_label_to_files[g_label].append(file)  
    
    positive_pairs = []
    for files in g_label_to_files.values():
        positive_pairs.extend((file1, file2) for file1, file2 in itertools.combinations(files, 2))

    # 记录正例对的数量
    num_positive_pairs = len(positive_pairs)

    max_dataset_size = num_positive_pairs * 2
    if  dataset_size > max_dataset_size:
        print('dataset_size is too large')
        sys.exit(1)

    negative_pairs = []
    labels = list(g_label_to_files.keys())
    for i, label_1 in enumerate(labels):
        for label_2 in labels[i+1:]:
            files_1 = g_label_to_files[label_1]
            files_2 = g_label_to_files[label_2]
            negative_pairs.extend((file1, file2) for file1, file2 in itertools.product(files_1, files_2))

    num_pairs_to_select = dataset_size // 2 

    selected_positive_pairs = random.sample(positive_pairs, num_pairs_to_select)
    selected_negative_pairs = random.sample(negative_pairs, num_pairs_to_select)

    train_size = int(num_pairs_to_select * 0.7)
    test_size = int(num_pairs_to_select * 0.2)

    train_pairs = {
        'positive': selected_positive_pairs[:train_size],
        'negative': selected_negative_pairs[:train_size]
    }
    test_pairs = {
        'positive': selected_positive_pairs[train_size:train_size + test_size],
        'negative': selected_negative_pairs[train_size:train_size + test_size]
    }
    valid_pairs = {
        'positive': selected_positive_pairs[train_size + test_size:],
        'negative': selected_negative_pairs[train_size + test_size:]
    }


    train_file = os.path.join(heteroG_dataset_dir, 'train.pth')
    test_file = os.path.join(heteroG_dataset_dir, 'test.pth')
    valid_file = os.path.join(heteroG_dataset_dir, 'valid.pth')

    torch.save(train_pairs, train_file)
    torch.save(test_pairs, test_file)
    torch.save(valid_pairs, valid_file)

if __name__=='__main__':

    debug=False
    small_dataset=True
    root=r'/home/ouyangchao/binsimgnn'
    save_dir=os.path.join(root,'dataset')
    dataset_size=5000  #最好是10的倍数
    random.seed(18)

    if debug:
        ll_file_dir=os.path.join(save_dir,'test_IR')
        heteroG_save_dir=os.path.join(save_dir,'test_heteroG')
        prep_log_dir=os.path.join(save_dir,'test_invalid_IR')
        prep_save_dir=os.path.join(save_dir,'test_preprocessed_IR')
        ir_programl_dir=os.path.join(save_dir,'test_ir_programl')
        vocab_dir=os.path.join(root,'test_vocab')
        heteroG_dataset_dir=os.path.join(root,'debug_heteroG_dataset')

    else:
        if small_dataset:
            ll_file_dir=os.path.join(save_dir,'binkit_small_IR')
            heteroG_save_dir=os.path.join(save_dir,'binkit_small_heteroG')
            prep_log_dir=os.path.join(save_dir,'binkit_small_invalid_IR')
            prep_save_dir=os.path.join(save_dir,'binkit_small_preprocessed_IR')
            ir_programl_dir=os.path.join(save_dir,'binkit_small_ir_programl')
            vocab_dir=os.path.join(root,'binkit_small_vocab')
            heteroG_dataset_dir=os.path.join(root,'binkit_small_heteroG_dataset')
 
        else:         
            print('please choose a dataset')   
            sys.exit(1)

    corpus_model_path=os.path.join(vocab_dir,'ir_corpus.model')
    corpus_vec_path=os.path.join(vocab_dir,'ir_corpus.vector')


    os.makedirs(heteroG_save_dir, exist_ok=True)
    os.makedirs(prep_log_dir, exist_ok=True)
    os.makedirs(prep_save_dir, exist_ok=True)
    os.makedirs(heteroG_dataset_dir,exist_ok=True)
    os.makedirs(ir_programl_dir,exist_ok=True)
    os.makedirs(vocab_dir,exist_ok=True)


    prep_ir_file(ll_file_dir,prep_save_dir,prep_log_dir)

    ll2programl(prep_save_dir,ir_programl_dir,heteroG_save_dir)

    programl2heteroG(ir_programl_dir)

    #归一化词向量
    for word in model.wv.key_to_index:
        model.wv[word] = normalize([model.wv[word]], norm='l2')[0]

    model.save(corpus_model_path)#.model可以用于继续训练
    model.wv.save_word2vec_format(corpus_vec_path)

    init_nodevector(heteroG_save_dir)

    build_dataset(heteroG_save_dir,heteroG_dataset_dir,dataset_size)
