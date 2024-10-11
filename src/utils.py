"""Data processing utilities."""

import json
import math
from texttable import Texttable
import torch
import re
from llvmlite import binding
import numpy as np
import os
from gensim.models import Word2Vec

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.set_cols_dtype(['t', 't'])  # 't' means text
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

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

def inst_node_isvalid(inst_node): #该指令是否有用
    if inst_node.features and inst_node.features.feature["full_text"].bytes_list.value :     
        inst_node_text = inst_node.features.feature["full_text"].bytes_list.value[0].decode('utf-8')
        if inst_node_text=='': #无用节点
            return False
        else:
            return True

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

def read_vector_from_file(file_path):
    with open(file_path, 'r') as file:
        vector_string = file.readline().strip()
        vector = np.array(list(map(float, vector_string.split())))
    return vector

class BatchProgramlCorpus:
    def __init__(self, ir_programl_list):
        self.ir_programl_list=ir_programl_list

    def __iter__(self):
        for ir_programl in self.ir_programl_list:
                for node in ir_programl.node:
                    if node.type==0:
                        if  inst_node_isvalid(node):#只收集有效inst节点的token
                            node_text = node.features.feature["full_text"].bytes_list.value[0].decode('utf-8')
                            node_text=normalize_inst(node_text)
                            node.text=node_text  #node.text 修改为归一化后的full_text
                            yield node_text.split()

                    elif node.type==3:
                        pass
                    else:
                        yield [node.text]

def gen_tokens(ir_programls_list):

    global model
    corpus = BatchProgramlCorpus(ir_programls_list)
    if len(model.wv) == 0:
        model.build_vocab(corpus)
    else:
        model.build_vocab(corpus, update=True)
        
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)


def collect_heteroG_files(root_dir):
    # 遍历所有子目录，收集所有的 .pth 文件路径
    #heteroG_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pth'):
                yield os.path.join(subdir,file)


def init_nodevector(ir_programl_dir,heteroG_save_dir):

    model = Word2Vec(vector_size=16, sg=0,window=8, min_count=1, workers=prop_threads) #参数需要做进一步实验看最好的效果
    #先生成vocab


    #为每个异构图初始化节点向量
    heteroG_files=collect_heteroG_files(heteroG_save_dir)
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


    #归一化词向量
    for word in model.wv.key_to_index:
        model.wv[word] = normalize([model.wv[word]], norm='l2')[0]

    model.save(corpus_model_path)#.model可以用于继续训练
    model.wv.save_word2vec_format(corpus_vec_path)