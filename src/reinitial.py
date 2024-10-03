import os
import torch


dataset_path=r'/home/ouyangchao/binsimgnn/binkit_small_heteroG_dataset'

def modify(data):
    delattr(data,'programl_graph')
    return data

def check_flow(data):
    for edge in data.programl_graph.edge:
        if edge.flow==3:
            print('test')
    
def check_dim(data):
    if data['inst', 'call', 'inst'].edge_index.ndim>2:
        print('test')

def find_pth_files(root_dir):
    pth_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pth'):
                pth_files.append(os.path.join(dirpath, filename))
    return pth_files

pth_files=find_pth_files(dataset_path)

for pth_file in pth_files:
    data = torch.load(pth_file)
    #data = modify(data)
    #torch.save(data, pth_file)
    #check_flow(data)
    check_dim(data)
print("所有 .pth 文件已处理完毕，")


