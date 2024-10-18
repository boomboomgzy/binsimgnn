import glob
import torch
import random
from tqdm import tqdm, trange
from torch import nn, optim
import torch.nn.functional as F
import datetime
from torch_geometric.nn import (
    HGTConv,
    GATConv,
    GINConv,
    SAGPooling,
    global_add_pool,
    Linear,
    Sequential
)
from torch_geometric.data import Batch,Data
import os
from SGFormer import TransConv
from utils import tab_printer

class DirHGTConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads,alpha):
        super(DirHGTConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.src_to_dst_metadata=(['data', 'inst'], [('inst', 'control', 'inst'), ('data', 'input', 'inst'), ('inst', 'output', 'data'), ('inst', 'call', 'inst')])
        self.dst_to_src_metadata=(['data', 'inst'], [('inst', 'be_control', 'inst'), ('inst', 'be_input', 'data'), ('data', 'be_output', 'inst'), ('inst', 'be_call', 'inst')])
        self.conv_src_to_dst = HGTConv(input_dim, output_dim, heads=heads,metadata=self.src_to_dst_metadata)
        self.conv_dst_to_src = HGTConv(input_dim, output_dim, heads=heads,metadata=self.dst_to_src_metadata)
        self.alpha = alpha

    def forward(self, x_dict, edge_index_dict):
        reversed_edge_index_dict={}
        for edge_type, edge_index in edge_index_dict.items():
            reversed_edge_index = edge_index[[1, 0], :]
            if edge_type==('inst', 'control', 'inst'):
                reversed_edge_index_dict[('inst', 'be_control', 'inst')] = reversed_edge_index
            elif edge_type==('data', 'input', 'inst'):
                reversed_edge_index_dict[('inst', 'be_input', 'data')] = reversed_edge_index
            elif edge_type==('inst', 'output', 'data'):
                reversed_edge_index_dict[('data', 'be_output', 'inst')] = reversed_edge_index
            else:
                reversed_edge_index_dict[('inst', 'be_call', 'inst')] = reversed_edge_index


        scaled_conv_src_to_dst_x_dict = {k: (1 - self.alpha) * v for k, v in self.conv_src_to_dst(x_dict,edge_index_dict).items()}
        scaled_conv_dst_to_src_x_dict = {k: self.alpha * v for k, v in self.conv_dst_to_src(x_dict,reversed_edge_index_dict).items()}


        return {k: scaled_conv_dst_to_src_x_dict[k] + scaled_conv_src_to_dst_x_dict[k] for k in scaled_conv_src_to_dst_x_dict.keys()}

class BinSimGNN(torch.nn.Module):

    def __init__(self, args):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(BinSimGNN, self).__init__()
        self.args = args
        self.setup_layers()
        self.init_parameters()



    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        #self.feature_count = self.args.tensor_neurons + self.args.bins
        self.feature_count=self.args.tensor_neurons

    def init_parameters(self):
        #torch.nn.init.xavier_uniform_(self.fully_connected_first.weight)
        #torch.nn.init.xavier_uniform_(self.scoring_layer.weight)
        pass

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.global_att = TransConv(in_channels=self.args.hidden_dim*self.args.heads,hidden_channels=self.args.hidden_dim*self.args.heads,dropout=self.args.dropout,num_heads=self.args.heads)
        self.pooling=SAGPooling(in_channels=self.args.hidden_dim*self.args.heads,ratio=0.5,GNN=GATConv)
        self.convs = torch.nn.ModuleList()

        #self.metadata=(['data', 'inst'], [('inst', 'control', 'inst'), ('data', 'input', 'inst'), ('inst', 'output', 'data'), ('inst', 'call', 'inst')])
        #self.metadata=(['data', 'inst'], [('inst', 'control', 'inst'), ('data', 'input', 'inst'), ('inst', 'output', 'data')])

        for _ in range(self.args.num_layers):
            conv=DirHGTConv(input_dim=-1,output_dim=self.args.hidden_dim*self.args.heads,heads=self.args.heads,alpha=0.5)
            #conv=HGTConv(in_channels=-1,out_channels=self.args.hidden_dim*self.args.heads,heads=self.args.heads,metadata=self.metadata)
            self.convs.append(conv)
        

        
    def conv_pass(self, x_dict,edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: torch.nn.functional.relu(x,inplace=True) for key, x in x_dict.items()}
            x_dict = {key: torch.nn.functional.dropout(x, p=self.args.dropout, training=self.training) for key, x in x_dict.items()}

        return x_dict

    def conv_pool_pass(self,batch_heteroG):

         for i,conv in enumerate(self.convs):
            #放入cuda
            batch_heteroG.x_dict = {key: tensor.cuda() for key, tensor in batch_heteroG.x_dict.items()}
            batch_heteroG.edge_index_dict = {key: tensor.cuda() for key, tensor in batch_heteroG.edge_index_dict.items()}
            
            x_dict = conv(batch_heteroG.x_dict, batch_heteroG.edge_index_dict)
            x_dict = {key: torch.nn.functional.relu(x,inplace=True) for key, x in x_dict.items()}
            x_dict = {key: torch.nn.functional.dropout(x, p=self.args.dropout, training=self.training) for key, x in x_dict.items()}   

            #更新异构图节点特征
            batch_heteroG['inst'].x=x_dict['inst']
            batch_heteroG['data'].x=x_dict['data']
            batch_heteroG.x_dict=x_dict

            #最后一层conv后不需要做池化
            if i != len(self.convs) - 1:
                #转换为同构图
                batch_homoG=batch_heteroG.to_homogeneous(edge_attrs=['edge_attr'])
                pooled_batch_homoG=Data()
                #做同构图池化
                pooled_batch_homoG.x, pooled_batch_homoG.edge_index, pooled_batch_homoG.edge_attr, pooled_batch_homoG.batch, perm, _ = self.pooling(batch_homoG.x.cuda(), batch_homoG.edge_index.cuda(), batch_homoG.edge_attr.cuda(), batch_homoG.batch.cuda())
                batch_homoG.x=batch_homoG.x.cpu()
                sg_batch_homoG=batch_homoG.subgraph(perm.cpu())
                sg_batch_homoG.x=pooled_batch_homoG.x.cpu()
                #得到新的池化后的异构图
                #batch_heteroG=sg_batch_homoG.to_heterogeneous(node_type=sg_batch_homoG.node_type,edge_type=sg_batch_homoG.edge_type,node_type_names=node_type_names,edge_type_names=edge_type_names)
                batch_heteroG=sg_batch_homoG.to_heterogeneous()
                x_dict = {}
                edge_index_dict={}
                for node_type in batch_heteroG.node_types:
                    x_dict[node_type] = batch_heteroG[node_type].x
                for edge_type in batch_heteroG.edge_types:
                    edge_index_dict[edge_type]=batch_heteroG[edge_type].edge_index

                batch_heteroG.x_dict = x_dict
                batch_heteroG.edge_index_dict=edge_index_dict

         return batch_heteroG

    def forward(self,batch_g1,batch_g2):


        #放入cuda
#        batch_g1.x_dict = {key: tensor.cuda() for key, tensor in batch_g1.x_dict.items()}
#        batch_g2.x_dict = {key: tensor.cuda() for key, tensor in batch_g2.x_dict.items()}
#        batch_g1.edge_index_dict = {key: tensor.cuda() for key, tensor in batch_g1.edge_index_dict.items()}
#        batch_g2.edge_index_dict = {key: tensor.cuda() for key, tensor in batch_g2.edge_index_dict.items()}

        batch_g1_conv = self.conv_pool_pass(batch_g1)
        batch_g2_conv = self.conv_pool_pass(batch_g2)


#        batch_homo_g1=batch_g1.to_homogeneous(edge_attrs=['edge_attr'])
#        batch_homo_g1.x, batch_homo_g1.edge_index, batch_homo_g1.edge_attr, batch_homo_g1.batch, _, _ = self.pooling(batch_homo_g1.x, batch_homo_g1.edge_index.cuda(), batch_homo_g1.edge_attr.cuda(), batch_homo_g1.batch.cuda())
#        batch_homo_g2=batch_g2.to_homogeneous(edge_attrs=['edge_attr'])
#        batch_homo_g2.x, batch_homo_g2.edge_index, batch_homo_g2.edge_attr, batch_homo_g2.batch, _, _ = self.pooling(batch_homo_g2.x, batch_homo_g2.edge_index.cuda(), batch_homo_g2.edge_attr.cuda(), batch_homo_g2.batch.cuda())
 
        batch_g1_conv_homo=batch_g1_conv.to_homogeneous()
        batch_g2_conv_homo=batch_g2_conv.to_homogeneous()

       #global_attention 
        batch_homo_g1_gatt_x=self.global_att(batch_g1_conv_homo)
        batch_homo_g2_gatt_x=self.global_att(batch_g2_conv_homo)

        batch_homo_g1_concat = torch.cat([batch_g1_conv_homo.x, batch_homo_g1_gatt_x], dim=1)
        batch_homo_g2_concat = torch.cat([batch_g2_conv_homo.x, batch_homo_g2_gatt_x], dim=1)

        batch_global_g1 = global_add_pool(batch_homo_g1_concat, batch_g1_conv_homo.batch.cuda())
        batch_global_g2 = global_add_pool(batch_homo_g2_concat, batch_g2_conv_homo.batch.cuda())

        cosine_similarities = F.cosine_similarity(batch_global_g1, batch_global_g2, dim=1)

         
        return cosine_similarities  #[batch_pairs_size]



     # x_dict 和 x 都要同时手动更新 
    def forward_old(self, batch_g1,batch_g2):

        batch_g1_x_dict = batch_g1.x_dict
        batch_g2_x_dict = batch_g2.x_dict
        batch_g1_edge_index_dict = batch_g1.edge_index_dict
        batch_g2_edge_index_dict = batch_g2.edge_index_dict

        batch_g1_x_dict = {key: tensor.cuda() for key, tensor in batch_g1_x_dict.items()}
        batch_g2_x_dict = {key: tensor.cuda() for key, tensor in batch_g2_x_dict.items()}
        batch_g1_edge_index_dict = {key: tensor.cuda() for key, tensor in batch_g1_edge_index_dict.items()}
        batch_g2_edge_index_dict = {key: tensor.cuda() for key, tensor in batch_g2_edge_index_dict.items()}

        batch_g1_x_dict_conv = self.conv_pass(batch_g1_x_dict, batch_g1_edge_index_dict)
        batch_g2_x_dict_conv = self.conv_pass(batch_g2_x_dict, batch_g2_edge_index_dict)

        #更新为卷积后的特征
        batch_g1['inst'].x=batch_g1_x_dict_conv['inst']
        batch_g1['data'].x=batch_g1_x_dict_conv['data']
        batch_g2['inst'].x=batch_g2_x_dict_conv['inst']
        batch_g2['data'].x=batch_g2_x_dict_conv['data']
        batch_g1.x_dict=batch_g1_x_dict_conv
        batch_g2.x_dict=batch_g2_x_dict_conv

        batch_homo_g1=batch_g1.to_homogeneous(edge_attrs=['edge_attr'])
        batch_homo_g1.x, batch_homo_g1.edge_index, batch_homo_g1.edge_attr, batch_homo_g1.batch, _, _ = self.pooling(batch_homo_g1.x, batch_homo_g1.edge_index.cuda(), batch_homo_g1.edge_attr.cuda(), batch_homo_g1.batch.cuda())
        batch_homo_g2=batch_g2.to_homogeneous(edge_attrs=['edge_attr'])
        batch_homo_g2.x, batch_homo_g2.edge_index, batch_homo_g2.edge_attr, batch_homo_g2.batch, _, _ = self.pooling(batch_homo_g2.x, batch_homo_g2.edge_index.cuda(), batch_homo_g2.edge_attr.cuda(), batch_homo_g2.batch.cuda())
 
       #global_attention 
        batch_homo_g1_gatt_x=self.global_att(batch_homo_g1)
        batch_homo_g2_gatt_x=self.global_att(batch_homo_g2)

        batch_homo_g1_concat = torch.cat([batch_homo_g1.x, batch_homo_g1_gatt_x], dim=1)
        batch_homo_g2_concat = torch.cat([batch_homo_g2.x, batch_homo_g2_gatt_x], dim=1)

        batch_global_g1 = global_add_pool(batch_homo_g1_concat, batch_homo_g1.batch)
        batch_global_g2 = global_add_pool(batch_homo_g2_concat, batch_homo_g2.batch)

        cosine_similarities = F.cosine_similarity(batch_global_g1, batch_global_g2, dim=1)

         
        return cosine_similarities  #[batch_pairs_size]


#        global_scores = self.tensor_network(batch_g1_global_x, batch_g2_global_x)  
#        
#        #得到节点所属的图索引
#        batch_indices_g1 = batch_g1_homo.batch  
#        batch_indices_g2 = batch_g2_homo.batch  
#
#        num_graphs = batch_indices_g1.max().item() + 1
#
#        hist_list=[]
#        for graph_idx in range(num_graphs):
#            node_mask_g1 = (batch_indices_g1 == graph_idx)
#            node_mask_g2 = (batch_indices_g2 == graph_idx)
#
#            # 提取当前图的节点特征
#            node_features_g1 = batch_g1_homo.x[node_mask_g1]
#            node_features_g2 = batch_g2_homo.x[node_mask_g2]
#
#            hist=self.calculate_histogram_blockwise(node_features_g1,node_features_g2)
#            hist_list.append(hist)
#
#
#        result_list = []
#
#        # 遍历 scores 和 hist 列表，进行逐个处理
#        for score, hist in zip(global_scores, hist_list):
#            score = score.view(1, -1)
#            score = torch.cat((score, hist), dim=1).view(1, -1)
#
#            score = torch.nn.functional.relu(self.fully_connected_first(score))
#            scaling_factor = 0.1  
#            score = score * scaling_factor  
#            score = torch.sigmoid(self.scoring_layer(score)) #值太大了 导致输出都是1
#
#            result_list.append(score)
#
#        return result_list



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, sim, labels):
        positive_loss = labels * (1-sim)**2
        negative_loss =  (1 - labels) * F.relu(sim - self.margin) ** 2
        loss = positive_loss + negative_loss
        return loss.mean()




class BinSimGNNTrainer(object):

    def __init__(self, args):

        self.args = args
        self.initial()
        self.setup_model()

    def setup_model(self):
        self.model = BinSimGNN(self.args)
        self.loss_fn = ContrastiveLoss(margin=0.1)

    def load_data(self):
        pass

    def initial(self):
        self.dataset_path=self.args.dataset
        self.train_g_pairs=torch.load(os.path.join(self.dataset_path,'train.pth'))
        self.test_g_pairs=torch.load(os.path.join(self.dataset_path,'test.pth'))
        self.valid_g_pairs=torch.load(os.path.join(self.dataset_path,'valid.pth'))


    def create_batches(self,g_pairs):
        batches = []
        positive_g_pairs = g_pairs['positive']
        negative_g_pairs = g_pairs['negative']
        all_pairs = positive_g_pairs + negative_g_pairs
        random.shuffle(all_pairs)

        batches_num = len(all_pairs) // self.args.batch_pairs_size

        for i in range(batches_num):
            batch = all_pairs[i * self.args.batch_pairs_size: (i + 1) * self.args.batch_pairs_size]
            batches.append(batch)

        remaining_pairs = all_pairs[batches_num * self.args.batch_pairs_size:]
        if remaining_pairs != []:
            batches.append(remaining_pairs)

        return batches
#
    def process_batch_g_pair(self, batch_g_pairs):
        self.optimizer.zero_grad()  

        batch_g1_list = []
        batch_g2_list = []
        batch_g_labels = []

        for g_pair in batch_g_pairs:
            g1 = torch.load(g_pair[0])
            g2 = torch.load(g_pair[1])
            batch_g1_list.append(g1)
            batch_g2_list.append(g2)
            batch_g_labels.append(1 if g1.g_label == g2.g_label else 0)

        batch_g_labels = torch.tensor(batch_g_labels, dtype=torch.float).cuda()

        batch_g1 = Batch.from_data_list(batch_g1_list,exclude_keys=['g_label','function_dict'])
        batch_g2 = Batch.from_data_list(batch_g2_list,exclude_keys=['g_label','function_dict'])
       
        batch_g_sim = self.model(batch_g1,batch_g2)  
        #batch_g_sim = torch.cat(batch_g_sim)
        batch_avg_loss_per_pair=self.loss_fn(batch_g_sim,batch_g_labels)
        batch_avg_loss_per_pair.backward()
       # for name, weight in self.model.named_parameters():
       #     if weight.requires_grad and weight.grad!=None:

       #         print('name ',name)
       #         print("weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())

        self.optimizer.step()

        return batch_avg_loss_per_pair.detach().item()


    def fit(self):
        """
        Fitting a model.
        """

        self.model = self.model.cuda()  
        
        print("\nModel training.\n")


        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        start_epoch=0

        best_metric = float('inf')  # 用于存储最佳模型的验证误差
        if self.args.load_path:
            load_dict=self.load()
            start_epoch=load_dict['epoch']+1
            best_metric=load_dict['metric']
            print('best_metrci: ',best_metric)

        self.model.train()
        #重新训练时超参数会被self.load覆盖 所有修改了的超参数需要再设置一遍
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.learning_rate        

        tab_printer(self.args)

        epochs = trange(start_epoch,self.args.epochs, leave=True, desc="Epoch")

        for epoch in epochs:

            batches = self.create_batches(self.train_g_pairs)
            epoch_loss_sum = 0 #统计这个epoch的总loss
            g_pairs_num = 0

            for batch_index, batch_g_pair in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                batch_avg_loss_per_pair = self.process_batch_g_pair(batch_g_pair) #得到batch中每对的平均loss
                g_pairs_num=g_pairs_num+len(batch_g_pair)
                epoch_loss_sum = epoch_loss_sum + batch_avg_loss_per_pair*len(batch_g_pair)
                epoch_avg_loss = epoch_loss_sum/g_pairs_num
                epochs.set_description("Epoch %d (Loss_Per_Pair=%g)" % (epoch,round(epoch_avg_loss, 10))) #得到当前每对的平均loss

            
            print(f"\nEpoch {epoch} completed, now evaluating on the validation set.")
            metric = self.score(mode='eval')  # 验证集评估
            print(f'eval metric: {str(round(metric, 10))}')

            if metric < best_metric:
                print(f"New best model found at epoch {epoch} with metric {str(round(metric, 10))}. Saving model.")
                best_metric=metric
                self.save(epoch,metric)
           
    def score(self,mode):
        """
        Scoring on the test set.
        """
        
        self.model = self.model.cuda()  

        if self.args.load_path:
            _=self.load()



        self.model.eval()
        self.ground_truth = []
        self.predition=[]

        if mode =='test':
            g_pairs=self.test_g_pairs
            print("\nFinal evaluation on the test set.")
        elif mode == 'eval':   
            g_pairs=self.valid_g_pairs
        else:
            print('error mode .please use test or eval')
            import sys
            sys.exit(1)

        loss_sum = 0 #统计总loss
        batches = self.create_batches(g_pairs)
        g_pairs_num = 0
        for index, batch_g_pair in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            batch_avg_loss_per_pair = self.process_batch_g_pair(batch_g_pair) #得到batch中每对的平均loss
            g_pairs_num=g_pairs_num+len(batch_g_pair)
            loss_sum = loss_sum + batch_avg_loss_per_pair*len(batch_g_pair)

        
        return loss_sum/g_pairs_num  #返回验证集/测试集中每对的平均loss

    def save(self,epoch,metric):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间格式: 年月日_时分秒

        filename = f"{current_time}_epoch={epoch}_metric={metric:.5f}.pth"
        
        save_path = f"{self.args.save_dir}/{filename}"
        checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epoch': epoch,
        'metric': metric
        }
        torch.save(checkpoint, save_path)
        print(f"Model and optimizer state saved to {save_path}.")

    def load(self):
        checkpoint = torch.load(self.args.load_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        rt={
            'epoch':checkpoint['epoch'],
            'metric':checkpoint['metric']
        }
        return rt

