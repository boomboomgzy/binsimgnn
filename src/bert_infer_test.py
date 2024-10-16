import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity

# 加载微调后的模型和tokenizer
model_save_path = r'/home/ouyangchao/binsimgnn/bert_test/codebert-llvm-ir-mlm/checkpoint-3153'  # 微调后模型的路径
model = RobertaModel.from_pretrained(model_save_path)
tokenizer = RobertaTokenizer.from_pretrained(model_save_path)

# 将模型设置为评估模式
model.eval()

similar_ir_pairs = [
    ("add %eax, %ebx", "add %edx, %ecx"),  # 相似的指令对
    ("sub %eax, %ebx", "sub %edx, %ecx"),
]

dissimilar_ir_pairs = [
    ("add %eax, %ebx", "mov %eax, %ebx"),  # 不相似的指令对
    ("sub %eax, %ebx", "mul %eax, %ebx"),
]

def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(**inputs)
    # 获取 [CLS] 标记的向量表示作为句向量
    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return sentence_embedding

# 处理一批IR指令对并生成句向量
def process_ir_pairs(ir_pairs):
    embeddings = []
    for ir1, ir2 in ir_pairs:
        emb1 = get_sentence_embedding(ir1)
        emb2 = get_sentence_embedding(ir2)
        embeddings.append((emb1, emb2))
    return embeddings

# 计算余弦相似度
def compute_cosine_similarity(embeddings):
    cos_similarities = []
    for emb1, emb2 in embeddings:
        emb1_np = emb1.cpu().numpy()  # 转为numpy数组
        emb2_np = emb2.cpu().numpy()
        cos_sim = cosine_similarity([emb1_np], [emb2_np])[0][0]  # 计算余弦相似度
        cos_similarities.append(cos_sim)
    return cos_similarities


# 生成相似和不相似句向量
similar_embeddings = process_ir_pairs(similar_ir_pairs)
dissimilar_embeddings = process_ir_pairs(dissimilar_ir_pairs)

# 计算相似和不相似的余弦相似度
similar_cos_sim = compute_cosine_similarity(similar_embeddings)
dissimilar_cos_sim = compute_cosine_similarity(dissimilar_embeddings)

print(similar_cos_sim)
print(dissimilar_cos_sim)