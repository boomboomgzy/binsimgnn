import torch    
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def exp_recall_mrr(testset_dir):
    recall_1_list=[]
    #recall_5_list=[]
    #recall_10_list=[]
    mrr_list=[]
    vec_list=[]
##    all 
#    for vec_file in os.listdir(testset_dir):
#            vec_file_path=os.path.join(testset_dir,vec_file)
#            vec_list.append(torch.load(vec_file_path).squeeze(0).numpy())
##
    
##   different optimization pair
#    opt_list=['O0','Ofast']
#    for vec_file in os.listdir(testset_dir):
#            vec_file_path=os.path.join(testset_dir,vec_file)
#            file_name=os.path.basename(vec_file_path)
#            match = re.search(r'_(O[0-3sfast]+)_', file_name)
#            if match:
#                opt_level = match.group(1)
#                if opt_level in opt_list:
#                    vec_list.append(torch.load(vec_file_path).squeeze(0).numpy())
#            else:
#                print("optimizetion not found")
#                return 
##   

##  different architectures pair
#    arch_list=['x86_64','mipseb_32']
#    for vec_file in os.listdir(testset_dir):
#            vec_file_path=os.path.join(testset_dir,vec_file)
#            file_name=os.path.basename(vec_file_path)
#            match = re.search(r'_(x86_(32|64)|arm_(32|64)|(mips|mipseb)_32)_', file_name)
#            if match:
#                arch = match.group(1)
#                if arch in arch_list:
#                    vec_list.append(torch.load(vec_file_path).squeeze(0).numpy())
#            else:
#                print("arch not found")
#                return 
##

## different compiler pair
    compiler_list=['gcc-11.2.0','clang-4.0']
    for vec_file in os.listdir(testset_dir):
            vec_file_path=os.path.join(testset_dir,vec_file)
            file_name=os.path.basename(vec_file_path)
            match = re.search(r'(gcc-\d+\.\d+\.\d+|clang-\d+\.\d+)', file_name)
            if match:
                compiler = match.group(1)
                if compiler in compiler_list:
                    vec_list.append(torch.load(vec_file_path).squeeze(0).numpy())
            else:
                print("compiler not found")
                return 
##

    vec_matrix=np.array(vec_list)
    simi_matrix=cosine_similarity(vec_matrix)
  
    for i in range(simi_matrix.shape[0]):
        # 排序每行的余弦相似度，索引越小表示越相似
        similar_indices = np.argsort(simi_matrix[i])[::-1]
        #top_10_simi=similar_indices[0:10]
        #top_5_simi=similar_indices[0:5]
        top_1_simi=similar_indices[0]
        rank = np.where(similar_indices == i)[0][0]  # 查找 i 的位置
        mrr_list.append(1/(rank + 1))


        if top_1_simi == i:
            recall_1_list.append(1)
        else:
            recall_1_list.append(0)
        #if i in top_5_simi:
        #     recall_5_list.append(1)
        #else:
        #     recall_5_list.append(0)
        #if i in top_10_simi:
        #     recall_10_list.append(1)
        #else:
        #     recall_10_list.append(0)


    recall_1_average = sum(recall_1_list) / len(recall_1_list)
    #recall_5_average = sum(recall_5_list) / len(recall_5_list)
    #recall_10_average = sum(recall_10_list) / len(recall_10_list)
    avg_mrr=sum(mrr_list) / len(mrr_list)
    
    print('avg mrr: ',avg_mrr)
    print(f'avg recall@1: {recall_1_average}')
    #print(f'avg recall@5: {recall_5_average}')
    #print(f'avg recall@10: {recall_10_average}')




def homoG_path2_ll_path(homoG_path):
        parts = list(homoG_path.parts)
        parts[parts.index("binkit_small_homoG")] = "binkit_small_preprocessed_IR"

        return Path(*parts).with_suffix('.ll')

def get_g_label(homoG_path):
    match=re.search(r'O[0-3sfast]+_(.*?)\.strip', homoG_path)
    bin_name=match.group(1)
    subdir=os.path.basename(os.path.dirname(homoG_path))
    return subdir+'_'+bin_name






if __name__=='__main__':
    testset_dir=r'/home/ouyangchao/binsimgnn/predit_result'
    exp_recall_mrr(testset_dir)