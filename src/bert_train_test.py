import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import programl
from utils import inst_node_isvalid,collect_programl_files

model_name = "microsoft/codebert-base-mlm"  # 可以换成其他适合的模型
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)

ir_programl_dir=r'/home/ouyangchao/binsimgnn/dataset/test_ir_programl'

save_dir=r'/home/ouyangchao/binsimgnn/bert_test'
model_save_path=os.path.join(save_dir,'codebert-llvm-ir-mlm')
log_path=os.path.join(save_dir,'logs')

def llvm_ir_token_generator(ir_programl_file_list):
    for ir_programl_file in ir_programl_file_list:
        ir_programl = programl.load_graphs(ir_programl_file)[0]
        for node in ir_programl.node:
            if node.type == 0:
                if inst_node_isvalid(node):  # 只收集有效inst节点的token
                    yield {"text": node.text}
            elif node.type == 3:
                pass
            else:
                yield {"text": node.text}  # 对其他类型节点处理


ir_programl_file_list=collect_programl_files(ir_programl_dir)

dataset = Dataset.from_generator(lambda: llvm_ir_token_generator(ir_programl_file_list))


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_test_split = tokenized_datasets.train_test_split(test_size=0.02,seed=42)

train_dataset = train_test_split['train']  
eval_dataset = train_test_split['test']  

# 使用 DataCollatorForLanguageModeling 生成掩码任务
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True,  # 开启掩码语言模型
    mlm_probability=0.15  # 掩码概率15%
)



##检查验证集的掩码
#example_batch = eval_dataset[:128]  
#
## 将样本数据转换为需要的格式
#inputs = {
#    'input_ids': torch.tensor(example_batch['input_ids']),  # 将 input_ids 转换为 tensor
#    'attention_mask': torch.tensor(example_batch['attention_mask'])  # attention mask 也需要传入
#}
#
##使用 DataCollatorForLanguageModeling 生成掩码后的数据
#masked_inputs = data_collator.torch_mask_tokens(inputs['input_ids'])
#masked_input_ids = masked_inputs[0]
#
#log_file=r'/home/ouyangchao/binsimgnn/bert_test/codebert-llvm-ir-mlm/check.log'
#with open(log_file, 'w') as f:
#    for i, input_ids in enumerate(masked_input_ids):
#        input_ids_list = input_ids.tolist()
#        decode_text=tokenizer.decode(input_ids_list, skip_special_tokens=False)
#        f.write(decode_text+'\n')

training_args = TrainingArguments(
    output_dir=model_save_path,
    overwrite_output_dir=True,
    num_train_epochs=3,  
    per_device_train_batch_size=150,
    per_device_eval_batch_size=150, 
    save_steps=500,  
    save_total_limit=2,  # 最多保存2个检查点
    learning_rate=2e-5,  # 学习率
    weight_decay=0.01,  # 权重衰减
    eval_strategy="steps",  # 每n步进行评估
    eval_steps=500,
    logging_dir=log_path,  # 日志文件夹
    log_level='info'
)

# 6. 使用Trainer进行模型训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 使用 tokenized 的训练集
    eval_dataset=eval_dataset,    # 使用 tokenized 的验证集
    data_collator=data_collator,  # 掩码任务
    tokenizer=tokenizer,
)
#继续训练
#resume_checkpoint=
#trainer.train(resume_from_checkpoint=resume_checkpoint)
trainer.train()


trainer.save_model(model_save_path)

