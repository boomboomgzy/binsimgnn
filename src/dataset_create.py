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

prop_threads=30



def programG2pyg(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[HeteroData, Iterable[HeteroData]]:

    def remove_metadata_and_comma(ir):
        #delete the meta info 
        pattern = r', ?!insn\.addr !\d+'
        import re
        return re.sub(pattern, '', ir)

    def _run_one(graph: ProgramGraph) -> HeteroData:
        # 4 lists, one per edge type
        # (control,input,output,call) control:inst->inst  input:data->inst output:inst->data call:inst->inst

        edge_index = [[], [], [], []] 
        edge_positions = [[], [], [], []]
        # 2 node type: data instruction
        data_nodes=[]
        inst_nodes=[]
        data_node_index_map={}
        inst_node_index_map={}

        function_dict={}
        for function in graph.function:
            function_dict[function.name]={'data':[],'inst':[]}


        for global_idx,node in enumerate(graph.node):
            if node.type==0:
                inst_nodes.append(node)
                inst_idx=len(inst_nodes)-1
                inst_node_index_map[str(global_idx)]=inst_idx
                function_dict[graph.function[node.function].name]['inst'].append(inst_idx)
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
            else: #inst->inst
                edge_index[3].append([inst_node_index_map[str(source_node)],inst_node_index_map[str(target_node)]])
                t=3
            #边和position一一对应
            edge_positions[t].append(edge.position)

        # Store the full text 
        data_node_text_list = []
        inst_node_text_list = []
        # Store the text and full text attributes
        for node in data_nodes:
            node_text=''

            if (
                node.features
                and node.features.feature["full_text"].bytes_list.value
            ):
                node_text = remove_metadata_and_comma(node.features.feature["full_text"].bytes_list.value[0].decode('utf-8'))

            data_node_text_list.append(node_text)

        for node in inst_nodes:
            node_text=''

            if (
                node.features
                and node.features.feature["full_text"].bytes_list.value
            ):
                node_text = remove_metadata_and_comma(node.features.feature["full_text"].bytes_list.value[0].decode('utf-8'))

            inst_node_text_list.append(node_text)

        # Pass from list to tensor
        edge_index = [torch.tensor(ej) for ej in edge_index]
        edge_positions = [torch.tensor(edge_pos_flow_type) for edge_pos_flow_type in edge_positions]


        # Create the graph structure
        hetero_graph = HeteroData()

        hetero_graph['data']['text'] = data_node_text_list
        hetero_graph['inst']['text'] = inst_node_text_list
        inst_feature_dim=32
        data_feature_dim=8
        hetero_graph['inst'].x=torch.rand((len(inst_nodes), inst_feature_dim), dtype=torch.float)
        hetero_graph['data'].x=torch.rand((len(data_nodes), data_feature_dim), dtype=torch.float)
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

        hetero_graph.function_dict=function_dict

        return hetero_graph

    if isinstance(graphs, ProgramGraph):
        return _run_one(graphs)

    return execute(_run_one, graphs, executor, chunksize)



def ir_file_preprocess(ll_file_path,log_dir,save_dir):
    ir_string=''
    with open(ll_file_path,'r') as file:
        ir_string=file.read()
        # 因llvm版本不一致导致的报错 暂且如此处理
        ir_string = ir_string.replace("%wide-string", "zeroinitializer")
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

def load_heteroG(heteroG_file_path):
    return torch.load(heteroG_file_path)

##def ir2vec_command(args):
##    ll_file_path,vec_file_path=args
##
##    command=[
##        ir2vec_path,
##        '-sym',
##        '-vocab',
##        vocabulary_path,
##        '-o',
##        vec_file_path,
##        '-level',
##        'p',
##        ll_file_path
##    ]
##    subprocess.run(command)


#ll_file_dir为预处理过的IR所在目录
#def getIRvec(ll_file_dir,vec_dir,ir2vec_path,vocabulary_path):
#    for subdir in os.listdir(ll_file_dir): #每个subdir 分开处理 
#        vec_file_path_list=[] #保存的vector文件
#        sub_dir_path = os.path.join(ll_file_dir, subdir)
#        ll_file_path_list=[] #当前subdir下所有ll_file
#        if os.path.isdir(sub_dir_path):
#            vec_sub_dir_path=os.path.join(vec_dir,subdir)
#            os.makedirs(vec_sub_dir_path,exist_ok=True)
#            for ll_file in os.listdir(sub_dir_path):
#                ll_file_path=os.path.join(sub_dir_path,ll_file)
#                if os.path.isfile(ll_file_path):
#                    ll_file_name=os.path.splitext(os.path.basename(ll_file_path))[0]
#                    ll_file_path_list.append(ll_file_path)
#                    vec_file_path_list.append(os.path.join(vec_sub_dir_path,ll_file_name+'.vec'))
#            
#
#            tasks=[]
#            for idx,ll_file_path in enumerate(ll_file_path_list):
#               tasks.append((ll_file_path,vec_file_path_list[idx]))
#
#            with multiprocessing.Pool(processes=prop_threads) as pool:
#                pool.map(ir2vec_command,tasks)



def read_vector_from_file(file_path):
    with open(file_path, 'r') as file:
        vector_string = file.readline().strip()
        vector = np.array(list(map(float, vector_string.split())))
    return vector



#ll_file_dir为预处理过的IR所在目录
def ll2heteroG(ll_file_dir,heteroG_save_dir):
    
    for subdir in os.listdir(ll_file_dir): #每个subdir 分开处理 
        heteroG_file_path_list=[] #保存的heteroG 的pth文件
        sub_dir_path = os.path.join(ll_file_dir, subdir)
        ll_file_path_list=[] #当前subdir下所有ll_file
        if os.path.isdir(sub_dir_path):
            heteroG_save_sub_dir_path=os.path.join(heteroG_save_dir,subdir)
            os.makedirs(heteroG_save_sub_dir_path,exist_ok=True)
            for ll_file in os.listdir(sub_dir_path):
                ll_file_path=os.path.join(sub_dir_path,ll_file)
                if os.path.isfile(ll_file_path):
                    ll_file_name=os.path.splitext(os.path.basename(ll_file_path))[0]
                    ll_file_path_list.append(ll_file_path)
                    heteroG_file_path_list.append(os.path.join(heteroG_save_sub_dir_path,ll_file_name+'.pth'))
        
        ll_ir_strings=[]
        ir_programls=None
        ir_heteroGs=None

        for ll_file_path in ll_file_path_list:
            with open(ll_file_path, 'r') as file:
                ll_ir_strings.append(file.read())



        with ThreadPoolExecutor(max_workers=prop_threads) as executor:
            ir_programls=list(programl.from_llvm_ir(ll_ir_strings, executor=executor,chunksize=prop_threads+1))


        with ThreadPoolExecutor(max_workers=prop_threads) as executor:
            ir_heteroGs=list(programG2pyg(ir_programls, executor=executor,chunksize=prop_threads+1))
        
        #subdir_binname  作为图的label 如果相同则相似 否则不相似
        for idx,ir_heteroG in enumerate(ir_heteroGs):
            ll_file_path=ll_file_path_list[idx]
            # 找到最后一个 "_" 的位置
            last_underscore_index = ll_file_path.rfind("_")

            # 找到 ".strip" 的位置
            strip_index = ll_file_path.find(".strip")
            bin_name=ll_file_path[last_underscore_index + 1:strip_index]
            subdir=os.path.basename(os.path.dirname(ll_file_path))
            ir_heteroG.g_label=subdir+'_'+bin_name

        for idx,heteroG_file_path in enumerate(heteroG_file_path_list):
            torch.save(ir_heteroGs[idx], heteroG_file_path)

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
    heteroG_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pth'):
                heteroG_files.append(os.path.join(subdir, file))
    return heteroG_files



def build_dataset(heteroG_save_dir,heteroG_dataset_dir):
    
    heteroG_files=collect_heteroG_files(heteroG_save_dir)
    random.shuffle(heteroG_files)
    train_ratio=0.8
    train_size = int(len(heteroG_files) * train_ratio)
    
    # 划分为训练集和测试集
    train_files = heteroG_files[:train_size]
    test_files = heteroG_files[train_size:]
    
    train_dir=os.path.join(heteroG_dataset_dir,'train')
    test_dir=os.path.join(heteroG_dataset_dir,'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for file in train_files:
        shutil.copy(file, train_dir)
    for file in test_files:
        shutil.copy(file, test_dir)



if __name__=='__main__':

    debug=False
    small_dataset=True
    save_dir=r'/home/ouyangchao/binsimgnn/dataset'

    if debug:
        ll_file_dir=os.path.join(save_dir,'test_IR')
        heteroG_save_dir=os.path.join(save_dir,'test_heteroG')
        prep_log_dir=os.path.join(save_dir,'test_invalid_IR')
        prep_save_dir=os.path.join(save_dir,'test_preprocessed_IR')
        #vec_dir=os.path.join(save_dir,'test_IR_vec')
        heteroG_dataset_dir=r'/home/ouyangchao/binsimgnn/debug_heteroG_dataset'

    else:
        if small_dataset:
            ll_file_dir=os.path.join(save_dir,'binkit_small_IR')
            heteroG_save_dir=os.path.join(save_dir,'binkit_small_heteroG')
            prep_log_dir=os.path.join(save_dir,'binkit_small_invalid_IR')
            prep_save_dir=os.path.join(save_dir,'binkit_small_preprocessed_IR')
            #vec_dir=os.path.join(save_dir,'binkit_small_IR_vec')
            heteroG_dataset_dir=r'/home/ouyangchao/binsimgnn/binkit_small_heteroG_dataset'
 
        else:            
            ll_file_dir=os.path.join(save_dir,'IR')
            heteroG_save_dir=os.path.join(save_dir,'heteroG')
            prep_log_dir=os.path.join(save_dir,'invalid_IR')
            prep_save_dir=os.path.join(save_dir,'preprocessed_IR')
            #vec_dir=os.path.join(save_dir,'IR_vec')
            heteroG_dataset_dir=r'/home/ouyangchao/binsimgnn/heteroG_dataset'
        
    os.makedirs(heteroG_save_dir, exist_ok=True)
    os.makedirs(prep_log_dir, exist_ok=True)
    os.makedirs(prep_save_dir, exist_ok=True)
    #os.makedirs(vec_dir, exist_ok=True)
    os.makedirs(heteroG_dataset_dir,exist_ok=True)

    #ir2vec_path=r'/home/ouyangchao/IR2Vec/build/bin/ir2vec'
    #vocabulary_path=r'/home/ouyangchao/IR2Vec/vocabulary/seedEmbeddingVocab-300-llvm8.txt'


    prep_ir_file(ll_file_dir,prep_save_dir,prep_log_dir)

    #getIRvec(prep_save_dir,vec_dir,ir2vec_path,vocabulary_path)

    #ll2heteroG(prep_save_dir,heteroG_save_dir,vec_dir)
    ll2heteroG(prep_save_dir,heteroG_save_dir)

    build_dataset(heteroG_save_dir,heteroG_dataset_dir)
