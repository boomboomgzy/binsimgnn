import torch
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import programl
from dataset_create import programG2pyg

import subprocess
import time
def getIRvec(ir2vec_path,ir_file_path,save_file_path,vocabulary_path='/home/ruan/gzy/IR2Vec/vocabulary/seedEmbeddingVocab-300-llvm8.txt'):
    command=[
        ir2vec_path,
        '-fa',
        '-vocab',
        vocabulary_path,
        '-o',
        save_file_path,
        '-level',
        'p',
        ir_file_path
    ]
    start_time = time.time()
    subprocess.run(command)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

import numpy as np

def read_vector_from_file(file_path):
    with open(file_path, 'r') as file:
        vector_string = file.readline().strip()
        vector = np.array(list(map(float, vector_string.split())))
    return vector

def cal_cossim(vec1,vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)



if __name__=='__main__':

    
    #check_path=r'/home/ruan/gzy/binsimgnn/dataset/test_heteroG/libtasn1/libtasn1-4.19.0_clang-4.0_arm_32_O1_asn1Coding.strip.preprocessed.pth'
    #heteroG=torch.load(check_path)
    ll_file_path=r'/home/ruan/gzy/binsimgnn/dataset/binkit_small_preprocessed_IR/cflow/cflow-1.7_clang-4.0_arm_32_O0_cflow.strip.preprocessed.ll'

    with open(ll_file_path, 'r') as file:
        ir_programl=programl.from_llvm_ir(file.read())
        ir_heteroG=programG2pyg(ir_programl)

    print('test')