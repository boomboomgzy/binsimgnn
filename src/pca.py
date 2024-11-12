import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import torch
import re

def get_vec(vec_dir):
    pth_files = [os.path.join(vec_dir, file) for file in os.listdir(vec_dir) if file.endswith('.pth')]
    vectors = []
    g_labels = []
    for pth_file in pth_files:
        match=re.search(r'O[0-3sfast]+_(.*?)\.pth', pth_file)
        bin_name=match.group(1)
        subdir=os.path.basename(pth_file).split('-')[0]
        g_label=subdir+'_'+bin_name
        vector = torch.load(pth_file)  
        vectors.append(vector.numpy())  
        g_labels.append(g_label)

    return np.vstack(vectors),g_labels

def pca_bin(vec_dir):

    X_v,y=get_vec(vec_dir)
    pca = PCA(n_components=2)
    X_p = pca.fit_transform(X_v)


    plt.figure(figsize=(18,12))
    plt.xlim(X_p[:, 0].min() * 1.5, X_p[:, 0].max() * 1.5)
    plt.ylim(X_p[:, 1].min() * 1.5, X_p[:, 1].max() * 1.5)

    colors = plt.get_cmap('tab20').colors

    for c, i, target_name in zip(colors, range(15), ['gdbm_gdbm_dump', 'gdbm_gdbmtool', 'gzip_gzip', 'readline_libhistory.so.8.2', 'cpio_cpio', 'gdbm_gdbm_load', 'hello_hello', 'libidn_idn2', 'cpio_rmt', 'gdbm_libgdbm.so.6.0.0', 'texinfo_install-info', 'gsasl_libgsasl.so.18.0.0', 'grep_grep', 'libidn_libidn2.so.0.3.8', 'readline_libreadline.so.8.2']):  
        plt.scatter(
            X_p[np.array(y) == target_name, 0], 
            X_p[np.array(y) == target_name, 1], 
            color=c, label=target_name, s=10, alpha=0.7, edgecolors='k'
        )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('64-Dimension Data PCA')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title='Categories')
    plt.savefig('/home/ouyangchao/binsimgnn/predit-pca.png', dpi=300)
    plt.close()

if __name__=='__main__':
    vec_dir=r'/home/ouyangchao/binsimgnn/predit_result'
    pca_bin(vec_dir)