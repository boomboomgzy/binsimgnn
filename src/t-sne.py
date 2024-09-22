import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

import os
import random
import numpy as np

def read_vector_from_file(file_path):
    with open(file_path, 'r') as file:
        vector_string = file.readline().strip()
        vector = np.array(list(map(float, vector_string.split())))
    return vector


main_dir = "/home/ouyangchao/binsimgnn/dataset/binkit_small_IR_vec"

# 用于存储所有的特征向量和对应的标签
X = []
y = []

label=0
label_name=[]
for sub_dir in os.listdir(main_dir):
    sub_dir_path = os.path.join(main_dir, sub_dir)
    
    if os.path.isdir(sub_dir_path):
        vector_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.vec')]
        
        selected_files = random.sample(vector_files, 10)
        
        for file in selected_files:
            file_path = os.path.join(sub_dir_path, file)
            
            vector = read_vector_from_file(file_path)
            
            X.append(vector)
            y.append(label)  

        label_name.append(sub_dir)
        label+=1

X = np.array(X)
y = np.array(y)

print(f"特征向量的维度：{X.shape}")
print(f"标签的维度：{y.shape}")




tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
      
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks()
plt.yticks()

plt.savefig('/home/ouyangchao/binsimgnn/tsne_result.png', dpi=300, bbox_inches='tight')

plt.close()