import torch
import os 
from utils import collect_homoG_files,inst_node_isvalid
import programl
import numpy as np
import subprocess
import concurrent.futures
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_shell_command(command):
    try:
        # 使用 subprocess.run 执行命令
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout  # 返回命令的输出
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"  # 返回错误信息



def gen_ir_vec_binkit(ll_dir,vocab_file_path,ir_vec_dir):
    max_threads = 30
    ll_file_paths=[]
    vec_json_file_paths=[]
    for root, dirs, files in os.walk(ll_dir):
        for file in files:
            ll_file_path=os.path.join(root, file)
            ll_file_paths.append(ll_file_path)
            sub_dir=os.path.basename(os.path.dirname(ll_file_path))
            vec_json_file_name = os.path.basename(ll_file_path)+'.vec.json'
            vec_json_file_path = os.path.join(ir_vec_dir,sub_dir,vec_json_file_name)
            os.makedirs(os.path.join(ir_vec_dir,sub_dir),exist_ok=True)
            with open(vec_json_file_path, 'w') as file:
                pass
            vec_json_file_paths.append(vec_json_file_path)


    commands = []
    for i,ll_file_path in enumerate(ll_file_paths):
        command=f'/home/ouyangchao/binkit_-ir2-vec/build/bin/ir2vec_binkit -sym_inst -vocab {vocab_file_path} -o {vec_json_file_paths[i]} {ll_file_path}'
        commands.append(command)


    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        executor.map(execute_shell_command, commands)



if __name__=="__main__":

    

   
   homoG_save_dir=r'/home/ouyangchao/binsimgnn/dataset/binkit_small_homoG_ir2vec'
   ir_programl_dir=r'/home/ouyangchao/binsimgnn/dataset/binkit_small_ir_programl'
   ll_dir=r'/home/ouyangchao/binsimgnn/dataset/binkit_small_preprocessed_IR'
   #ll_dir=r'/home/ouyangchao/binsimgnn/dataset/test_preprocessed_IR'
   vocab_file_path=r'/home/ouyangchao/binkit_-ir2-vec/seed_embeddings/seedEmbedding_1500E_64D.txt'
   ir_vec_dir=r'/home/ouyangchao/binsimgnn/binkit_ir_vec'

   #gen_ir_vec_binkit(ll_dir,vocab_file_path,ir_vec_dir)



   homoG_files=collect_homoG_files(homoG_save_dir)

   def process_file(homoG_file):
        homoG=torch.load(homoG_file)
        programl_file_sub_dir=os.path.basename(os.path.dirname(homoG_file))
        programl_file_name = os.path.basename(homoG_file).replace('.pth', '.prog')
        ll_file_name=os.path.basename(homoG_file).replace('.pth', '.ll')
        programl_file_path=os.path.join(ir_programl_dir,programl_file_sub_dir,programl_file_name)
        ll_file_path=os.path.join(ll_dir,programl_file_sub_dir,ll_file_name)
        programl_graph=programl.load_graphs(programl_file_path)[0]
        vec_file_path=os.path.join(ir_vec_dir,programl_file_sub_dir,ll_file_name+'.vec.json')
               
        with open(vec_file_path, 'r') as file:
            inst_vec_dict = json.load(file)

            inst_nodes = []
            inst_features = []

            for node in programl_graph.node:
                if node.type == 0 and inst_node_isvalid(node):
                    inst_nodes.append(node)

            for node in inst_nodes:
                if node.text == '[external]':
                    float_list = [0.01] * 64
                else:
                    inst = node.features.feature["full_text"].bytes_list.value[0].decode('utf-8')
                    float_list = inst_vec_dict[inst.strip()]
                inst_features.append(float_list)

            # 转换为张量并保存
            homoG.x = torch.tensor(np.array(inst_features), dtype=torch.float)
            torch.save(homoG, homoG_file)  

   threads_num=30
   with ThreadPoolExecutor(max_workers=threads_num) as executor:
       futures = [executor.submit(process_file, file) for file in homoG_files]
       for future in as_completed(futures):
           try:
               future.result()
           except Exception as e:
               print(f"Error processing file: {e}")