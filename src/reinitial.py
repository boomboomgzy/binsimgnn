import os
import torch


dataset_path=r'/home/ouyangchao/binsimgnn/binkit_small_heteroG_dataset'

def generate_new_features(data):
    inst_feature_dim=32
    data_feature_dim=8
    data['inst'].x=torch.rand((len(data['inst'].text), inst_feature_dim), dtype=torch.float)
    data['data'].x=torch.rand((len(data['data'].text), data_feature_dim), dtype=torch.float)
    return data

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
    data = generate_new_features(data)
    torch.save(data, pth_file)

print("所有 .pth 文件已处理完毕，节点特征已更新。")


