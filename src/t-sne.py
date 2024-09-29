import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np



def read_token_vec(token_file):
    token_vecs={}
    with open(token_file, 'r') as file:
        for line in file:
            line = line.strip()  
            key, value = line.split(" : ")
            token_vecs[key] = np.array([float(x) for x in value.split(",")])
    return token_vecs



def tsne_token(token_file):
    token_dict=read_token_vec(token_file)
    keys = list(token_dict.keys())
    values = np.array(list(token_dict.values()))

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(values)

    plt.figure(figsize=(100, 80))
    for i, key in enumerate(keys):
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], label=key)

    plt.title('t-SNE Visualization of tokens')
    plt.legend()
    plt.savefig('/home/ouyangchao/binsimgnn/tsne_token.png', dpi=100, bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    token_file=r'/home/ouyangchao/binsimgnn/tokenvec.txt'
    tsne_token(token_file)
