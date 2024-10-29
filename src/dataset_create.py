import programl
import json
import os
import torch
from torch_geometric.data import Data
from programl.proto import ProgramGraph
from typing import Iterable, Optional, Union
from programl.util.py.executor import ExecutorLike, execute
from concurrent.futures import ThreadPoolExecutor
import sys
import random
import re
from programl.proto import program_graph_pb2
from collections import defaultdict
import itertools
from utils import remove_metadata_comma_align,ir_validation,inst_node_isvalid,collect_homoG_files,normalize_inst

prop_threads=20 #不要开太大  不然内存占用会偏大


homoG_edge_type_dict={
    'call':0,
    'control':1
}
homoG_edge_type_count=2

def programG2pyg(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> None:

    #homoG version
    def _run_one(graph: ProgramGraph) -> None:

            # 4 lists, one per edge type
        # (control,input,output,call) control:inst->inst  input:data->inst output:inst->data   call:inst->inst
            edge_index = [[], [], [],[]] 
            #edge_positions = [[], [], [],[]]
            # 2 node type: data instruction
            data_nodes=[]
            inst_nodes=[]
            data_node_index_map={}
            inst_node_index_map={}
            unvalid_inst_nodes_global_idx=[]


            for global_idx,node in enumerate(graph.node):
                if node.type==0:
                    if inst_node_isvalid(node):
                        inst_nodes.append(node)
                        inst_idx=len(inst_nodes)-1
                        inst_node_index_map[str(global_idx)]=inst_idx
                    else:
                        unvalid_inst_nodes_global_idx.append(global_idx)
                elif node.type==3:
                    pass
                else:
                    data_nodes.append(node)
                    data_idx=len(data_nodes)-1
                    data_node_index_map[str(global_idx)]=data_idx


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
                #edge_positions[t].append(edge.position)


            # Pass from list to tensor

            edge_index = [torch.tensor(ej) for ej in edge_index]

            #创建同构图  inst->inst 使用数字编号来区分边类型
            global homoG_edge_type_count

            # Create the graph structure
            inst_control_inst_edge_index=edge_index[0].T.contiguous()            
            data_input_inst_edge_index=edge_index[1].T.contiguous()            
            inst_output_data_edge_index=edge_index[2].T.contiguous()            
            inst_call_inst_edge_index=edge_index[3].T.contiguous()            

            #边索引去重处理
            inst_control_inst_edge_index = torch.unique(inst_control_inst_edge_index.T, dim=0).T
            data_input_inst_edge_index = torch.unique(data_input_inst_edge_index.T, dim=0).T
            inst_output_data_edge_index = torch.unique(inst_output_data_edge_index.T, dim=0).T
            inst_call_inst_edge_index = torch.unique(inst_call_inst_edge_index.T, dim=0).T


            homoG_edge_index=[]
            homoG_edge_attr=[]

            #contral edge and call edge  add directly
            homoG_edge_index.append(inst_control_inst_edge_index)
            homoG_edge_index.append(inst_call_inst_edge_index)
            homoG_edge_attr.extend([1] * inst_control_inst_edge_index.shape[1])  # 控制流边
            homoG_edge_attr.extend([0] * inst_call_inst_edge_index.shape[1])    # 调用边

            data_to_inst_dict = {}
            for i in range(data_input_inst_edge_index.size(1)):
                data_idx = data_input_inst_edge_index[0, i].item()  
                inst_idx = data_input_inst_edge_index[1, i].item()  
                if data_idx not in data_to_inst_dict:
                    data_to_inst_dict[data_idx] = []
                data_to_inst_dict[data_idx].append(inst_idx)  # 记录该data节点指向的所有inst节点

            matched_data_inst_edges = set()
            #unmatched_data_inst_edges = set()

            inst_to_inst_dataflow_edge_index=[]
            for i in range(inst_output_data_edge_index.size(1)):
                data_idx = inst_output_data_edge_index[1, i].item()  
                inst_idx = inst_output_data_edge_index[0, i].item()  
                #data=
                if data_idx in data_to_inst_dict:
                    # 若找到相应的 data -> inst，生成 inst -> inst 边
                    for target_inst_idx in data_to_inst_dict[data_idx]:
                        inst_to_inst_dataflow_edge_index.append([inst_idx, target_inst_idx])
                        matched_data_inst_edges.add((data_idx,target_inst_idx))
                        

                else:
                    # 未找到相应的 data -> inst,  这条inst->data边是孤立的
                    #则 inst->external
                        inst_to_inst_dataflow_edge_index.append([inst_idx, 0])

                #添加这条边的边属性        
                edge_type=data_nodes[data_idx].text
                if edge_type in homoG_edge_type_dict:
                    edge_value = homoG_edge_type_dict[edge_type]
                else:
                    homoG_edge_type_dict[edge_type] = homoG_edge_type_count
                    edge_value = homoG_edge_type_count
                    homoG_edge_type_count += 1
            
                homoG_edge_attr.append(edge_value)           
        
        
        
        #data_input_inst_edge_index_set=set()

            for i in range(data_input_inst_edge_index.size(1)):
                data_idx = data_input_inst_edge_index[0, i].item()
                inst_idx = data_input_inst_edge_index[1, i].item()
                edge = (data_idx, inst_idx)
                #data_input_inst_edge_index_set.add(edge)
                # 检查该边是否在 matched_edges_set 中
                if edge not in matched_data_inst_edges:
                    #处理 孤立的 data->inst 边
                    #unmatched_data_inst_edges.add(edge)
                    inst_to_inst_dataflow_edge_index.append([0,inst_idx])
                    edge_type=data_nodes[data_idx].text
                    if edge_type in homoG_edge_type_dict:
                        edge_value = homoG_edge_type_dict[edge_type]
                    else:
                        homoG_edge_type_dict[edge_type] = homoG_edge_type_count
                        edge_value = homoG_edge_type_count
                        homoG_edge_type_count += 1

                    homoG_edge_attr.append(edge_value)



            inst_to_inst_dataflow_edge_index=torch.tensor(inst_to_inst_dataflow_edge_index)        
            homoG_edge_index.append(inst_to_inst_dataflow_edge_index.T.contiguous())
            homoG_edge_index = torch.cat(homoG_edge_index, dim=1)
            homoG_edge_attr=torch.tensor(homoG_edge_attr)

            # 构建同构图
            homoG = Data()
            homoG.edge_index = homoG_edge_index
            homoG.edge_attr = homoG_edge_attr

            #subdir_binname  作为图的label 如果相同则相似 否则不相似
            homoG_file_path=graph.module[-1].name
            match=re.search(r'O[0-3sfast]+_(.*?)\.strip', homoG_file_path)
            bin_name=match.group(1)
            subdir=os.path.basename(os.path.dirname(homoG_file_path))
            homoG.g_label=subdir+'_'+bin_name
            
            #保存homoG
            homoG_subdir_path=os.path.dirname(homoG_file_path)
            os.makedirs(homoG_subdir_path,exist_ok=True)

            torch.save(homoG,homoG_file_path)

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




#预处理后的IR
def ll2programl(ll_file_dir,ir_programl_dir,homoG_save_dir):

    for subdir in os.listdir(ll_file_dir): 
        homoG_file_path_list=[] #保存的homoG 的pth文件
        sub_dir_path = os.path.join(ll_file_dir, subdir)
        ll_file_path_list=[] #当前subdir下所有ll_file
        homoG_save_sub_dir_path=os.path.join(homoG_save_dir,subdir)
        os.makedirs(homoG_save_sub_dir_path,exist_ok=True)
        for ll_file in os.listdir(sub_dir_path):
            ll_file_path=os.path.join(sub_dir_path,ll_file)
            ll_file_name=os.path.splitext(os.path.basename(ll_file_path))[0]
            ll_file_path_list.append(ll_file_path)
            homoG_file_path_list.append(os.path.join(homoG_save_sub_dir_path,ll_file_name+'.pth'))

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
        #直接在这里对指令做正则化,后面无需再做
        for id,ir_prog in enumerate(ir_programls):
            for node in ir_prog.node:
                if node.type==0:
                    if inst_node_isvalid(node):#只收集有效inst节点的token
                        if node.text!='[external]':
                            node_text = node.features.feature["full_text"].bytes_list.value[0].decode('utf-8')
                            node_text=normalize_inst(node_text)
                            node.text=node_text  #node.text 修改为归一化后的full_text 
            
            new_module = program_graph_pb2.Module()
            new_module.name = homoG_file_path_list[id]
            ir_prog.module.append(new_module)
            file_name=os.path.splitext(os.path.basename(homoG_file_path_list[id]))[0]
            prog_save_path=os.path.join(prog_save_sub_dir,file_name+'.prog')

            programl.save_graphs(prog_save_path,[ir_prog])


#ll_file_dir为预处理过的IR所在目录
def programl2homoG(ir_programl_dir):

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




def build_dataset(homoG_save_dir,homoG_dataset_dir,dataset_size):
    
    homoG_files=collect_homoG_files(homoG_save_dir)



    g_label_to_files = defaultdict(list)
    for file in homoG_files:
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


    train_file = os.path.join(homoG_dataset_dir, 'train.pth')
    test_file = os.path.join(homoG_dataset_dir, 'test.pth')
    valid_file = os.path.join(homoG_dataset_dir, 'valid.pth')

    torch.save(train_pairs, train_file)
    torch.save(test_pairs, test_file)
    torch.save(valid_pairs, valid_file)

if __name__=='__main__':

    debug=False
    small_dataset=True
    root=r'/home/ouyangchao/binsimgnn'
    save_dir=os.path.join(root,'dataset')
    dataset_size=5000  #最好是10的倍数
    #dataset_size=20
    random.seed(18)

    if debug:
        ll_file_dir=os.path.join(save_dir,'test_IR')
        homoG_save_dir=os.path.join(save_dir,'test_homoG')
        prep_log_dir=os.path.join(save_dir,'test_invalid_IR')
        prep_save_dir=os.path.join(save_dir,'test_preprocessed_IR')
        ir_programl_dir=os.path.join(save_dir,'test_ir_programl')
        homoG_dataset_dir=os.path.join(root,'debug_homoG_dataset')
        edge_type_file_path=os.path.join(save_dir,'debug_homoG_edge_type.json')

    else:
        if small_dataset:
            ll_file_dir=os.path.join(save_dir,'binkit_small_IR')
            homoG_save_dir=os.path.join(save_dir,'binkit_small_homoG')
            prep_log_dir=os.path.join(save_dir,'binkit_small_invalid_IR')
            prep_save_dir=os.path.join(save_dir,'binkit_small_preprocessed_IR')
            ir_programl_dir=os.path.join(save_dir,'binkit_small_ir_programl')
            homoG_dataset_dir=os.path.join(root,'binkit_small_homoG_dataset')
            edge_type_file_path=os.path.join(save_dir,'binkit_small_homoG_edge_type.json')
        else:         
            print('please choose a dataset')   
            sys.exit(1)




    os.makedirs(homoG_save_dir, exist_ok=True)
    os.makedirs(prep_log_dir, exist_ok=True)
    os.makedirs(prep_save_dir, exist_ok=True)
    os.makedirs(homoG_dataset_dir,exist_ok=True)
    os.makedirs(ir_programl_dir,exist_ok=True)


    #prep_ir_file(ll_file_dir,prep_save_dir,prep_log_dir)

    #ll2programl(prep_save_dir,ir_programl_dir,homoG_save_dir)

    programl2homoG(ir_programl_dir)

    #保存边类型词典
    with open(edge_type_file_path, 'w') as file:
        json.dump(homoG_edge_type_dict, file)

    build_dataset(homoG_save_dir,homoG_dataset_dir,dataset_size)
