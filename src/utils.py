
import programl
from texttable import Texttable
import torch
import re
from llvmlite import binding
import numpy as np
import os
from gensim.models import Word2Vec,FastText
from sklearn.preprocessing import normalize
from tqdm import tqdm


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
    #该节点要特殊处理
    if inst_node.text=='[external]':
        return True
    if inst_node.features and inst_node.features.feature["full_text"].bytes_list.value :     
        inst_node_text = inst_node.features.feature["full_text"].bytes_list.value[0].decode('utf-8')
        if inst_node_text=='': #无用节点
            return False
        else:
            return True

def normalize_inst(inst):
    inst=re.sub(r'@global_var_\w+', '@global_var', inst)
    inst=re.sub(r'@\d+', '@global_var', inst)
    inst=re.sub(r'%[\w\.\-]+', '%ID', inst)
    inst=re.sub(r'(?<=\s)-?\d+(\.\d+)?\b', '%CONST', inst)
    inst = re.sub(r'@function_\w+', '@function', inst)
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
    def __init__(self, ir_programl_file_list):
        self.ir_programl_file_list=ir_programl_file_list

    def __iter__(self):
        for ir_programl_file in self.ir_programl_file_list:
                ir_programl=programl.load_graphs(ir_programl_file)[0]
                for node in ir_programl.node:
                    if node.type==0:
                        if inst_node_isvalid(node):#只收集有效inst节点的token
                                yield node.text.split()
                    elif node.type==3:
                        pass
                    else:
                        yield [node.text]
                


def collect_heteroG_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pth'):
                yield os.path.join(subdir,file)


def collect_programl_files(root_dir):
    ir_programl_file_list=[]
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.prog'):
                ir_programl_file_list.append(os.path.join(subdir,file))
    
    return ir_programl_file_list



def init_nodevector(ir_programl_dir,heteroG_save_dir,corpus_model_path,corpus_vec_path,prop_threads,vector_log_file):

    #先生成vocab
    model = FastText(
    vector_size=64,   
    window=10,             
    min_count=1,          
    epochs=5,            
    min_n=2,              
    max_n=6,             
    word_ngrams=1,
    workers=prop_threads
    )
    #model = Word2Vec(vector_size=64, sg=1,negative=10, window=10, min_count=1, workers=prop_threads,epochs=10,alpha=0.05,min_alpha=0.001,sample=1e-4) #参数需要做进一步实验看最好的效果
    corpus = BatchProgramlCorpus(collect_programl_files(ir_programl_dir))
    model.build_vocab(corpus)
    #model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    with tqdm(total=model.epochs, desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(model.epochs):
            model.train(corpus, total_examples=model.corpus_count, epochs=1) 
            pbar.update(1)  # 更新进度条


    #归一化词向量
    #for word in model.wv.key_to_index:
    #    model.wv[word] = normalize([model.wv[word]], norm='l2')[0]

    model.save(corpus_model_path)#.model可以用于继续训练
    model.wv.save_word2vec_format(corpus_vec_path)

    #这里可以多线程
    #为每个异构图初始化节点向量
    heteroG_files=collect_heteroG_files(heteroG_save_dir)
    for heteroG_file in heteroG_files:
        heteroG=torch.load(heteroG_file)
        programl_file_sub_dir=os.path.basename(os.path.dirname(heteroG_file))
        programl_file_name = os.path.basename(heteroG_file).replace('.pth', '.prog')
        programl_file_path=os.path.join(ir_programl_dir,programl_file_sub_dir,programl_file_name)
        programl_graph=programl.load_graphs(programl_file_path)[0]
        
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

        #记录每个节点的向量
        v_file=open(vector_log_file,'w') 

        for node in inst_nodes:
            inst=node.text
            token_list=inst.split()
            token_vectors = [model.wv[token] for token in token_list]
            node_vector=np.sum(token_vectors, axis=0)
            #node_vector=np.mean(token_vectors,axis=0)
            inst_features.append(node_vector)
            str_node_vector='['+','.join(map(str, node_vector))+']'
            v_file.write(f"{inst} : {str_node_vector}\n") 

        for node in data_nodes:
            data=node.text
            node_vector=model.wv[data]
            data_features.append(node_vector)
            str_node_vector='['+','.join(map(str, node_vector))+']'
            v_file.write(f"{data} : {str_node_vector}\n")

        v_file.close()
        # x_dict 和 x 最后都要同时手动更新 
        heteroG['inst'].x= torch.tensor(np.array(inst_features), dtype=torch.float)
        heteroG['data'].x= torch.tensor(np.array(data_features), dtype=torch.float)
        heteroG.x_dict={
            'inst': heteroG['inst'].x,
            'data': heteroG['data'].x
            }

        #初始化完后可以删除heteroG.programl_graph
        if hasattr(heteroG, 'programl_graph'):
            delattr(heteroG, 'programl_graph')

        torch.save(heteroG, heteroG_file)


