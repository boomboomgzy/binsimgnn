import IR2Vec
import torch    
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import re
import os


torch.cuda.set_device(3)

def homoG_path2_ll_path(homoG_path):
        parts = list(homoG_path.parts)
        parts[parts.index("binkit_small_homoG")] = "binkit_small_preprocessed_IR"

        return Path(*parts).with_suffix('.ll')

def get_g_label(homoG_path):
    match=re.search(r'O[0-3sfast]+_(.*?)\.strip', homoG_path)
    bin_name=match.group(1)
    subdir=os.path.basename(os.path.dirname(homoG_path))
    return subdir+'_'+bin_name





def ir2vec_test(record_file_path):    
    # 加载测试集
    test_g_pairs = torch.load('/home/ouyangchao/binsimgnn/binkit_small_homoG_dataset/test.pth')
    positive_g_pairs = test_g_pairs['positive']
    negative_g_pairs = test_g_pairs['negative']
    all_pairs = positive_g_pairs + negative_g_pairs

    record_file=open(record_file_path,'w')
    for g_pair in tqdm(all_pairs):
        g_label = 1 if get_g_label(g_pair[0]) == get_g_label(g_pair[1]) else 0

        g1_path = homoG_path2_ll_path(Path(g_pair[0]))
        g2_path = homoG_path2_ll_path(Path(g_pair[1]))
        g1_emb = IR2Vec.generateEmbeddings(str(g1_path), "fa", "p")['Program_List']
        g2_emb = IR2Vec.generateEmbeddings(str(g2_path), "fa", "p")['Program_List']
        g1_emb = torch.tensor(g1_emb, dtype=torch.float32).cuda()
        g2_emb = torch.tensor(g2_emb, dtype=torch.float32).cuda()
        cosine_similarities = F.cosine_similarity(g1_emb.unsqueeze(0), g2_emb.unsqueeze(0))

        record_file.write(f"{g_pair[0]} - {g_pair[1]} ,label = {g_label} : Similarity = {cosine_similarities.item()}\n")

    record_file.close()

if __name__=='__main__':
    test_record_file_path='/home/ouyangchao/binsimgnn/ir2vec_test_record.txt'
    ir2vec_test(test_record_file_path)