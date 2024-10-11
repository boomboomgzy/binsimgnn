from dataset_create import inst_node_isvalid,normalize_inst
from gensim.models import Word2Vec

prop_threads=10

model = Word2Vec(vector_size=16, sg=0,window=8, min_count=1, workers=prop_threads) #参数需要做进一步实验看最好的效果






