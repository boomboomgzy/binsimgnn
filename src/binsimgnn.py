import glob
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from layers import AttentionModule, TenorNetworkModule
from utils import calculate_loss, calculate_normalized_ged,calculate_cossim
from torch import nn, optim
import torch.nn.functional as F
import datetime
from torch_geometric.nn import (
    HGTConv,
)
import torch.nn as nn


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

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons + self.args.bins
        
    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()

        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)

        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        self.normalize = False
        self.convs = torch.nn.ModuleList()


        for _ in range(self.args.num_layers):
            conv=DirHGTConv(input_dim=-1,output_dim=self.args.hidden_dim*self.args.heads,heads=self.args.heads,alpha=0.5)
            self.convs.append(conv)

    def calculate_histogram(self, f1_x, f2_x):
        """
        Calculate histogram from similarity matrix.
        :param abstract_featfeatures_2: Feature matrix for graph 2.
        :return hiures_1: Feature matrix for graph 1.
        :param abstract_st: Histsogram of similarity scores.
        """
    
        scores = torch.mm(f1_x,torch.t(f2_x)).detach()   
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        
        return hist

    def get_function_x(self,g_function_dict,g_x_dict):
        function_x = []
        for _, node_dict in g_function_dict.items():
            data_x = torch.empty(0, self.args.hidden_dim*self.args.heads,dtype=torch.float).cuda()
            inst_x = torch.empty(0, self.args.hidden_dim*self.args.heads,dtype=torch.float).cuda()
            #data_x = torch.empty(0, self.args.hidden_dim*self.args.heads,dtype=torch.float)
            #inst_x = torch.empty(0, self.args.hidden_dim*self.args.heads,dtype=torch.float)

            if len(node_dict['data'])>0:
                data_x = torch.stack([g_x_dict['data'][data_node_id] for data_node_id in node_dict['data']])

            if len(node_dict['inst'])>0:
                inst_x = torch.stack([g_x_dict['inst'][inst_node_id] for inst_node_id in node_dict['inst']])
            
            combined_x = torch.cat((data_x, inst_x), dim=0)
            pooled_x = self.attention(combined_x) #函数的特征矩阵送入att层  得到函数表示
            function_x.append(pooled_x.squeeze())        

        return torch.stack(function_x)

    def convolutional_pass(self, x_dict,edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: torch.nn.functional.relu(x,inplace=True) for key, x in x_dict.items()}
            x_dict = {key: torch.nn.functional.dropout(x, p=self.args.dropout, training=self.training) for key, x in x_dict.items()}

        return x_dict

    def forward(self, x_dict,edge_index_dict,function_dict):
        
        g1_x_dict,g2_x_dict=x_dict
        g1_edge_index_dict,g2_edge_index_dict=edge_index_dict
        g1_function_dict,g2_function_dict=function_dict



        g1_x_dict = {key: tensor.cuda() for key, tensor in g1_x_dict.items()}
        g2_x_dict = {key: tensor.cuda() for key, tensor in g2_x_dict.items()}

        g1_edge_index_dict = {key: tensor.cuda() for key, tensor in g1_edge_index_dict.items()}
        g2_edge_index_dict = {key: tensor.cuda() for key, tensor in g2_edge_index_dict.items()}

       ##g1_x_dict = {key: tensor for key, tensor in g1_x_dict.items()}
       ##g2_x_dict = {key: tensor for key, tensor in g2_x_dict.items()}

       ##g1_edge_index_dict = {key: tensor for key, tensor in g1_edge_index_dict.items()}
       ##g2_edge_index_dict = {key: tensor for key, tensor in g2_edge_index_dict.items()}

        #GCN层提取节点特征

        g1_x_dict_conv = self.convolutional_pass(g1_x_dict,g1_edge_index_dict)
        g2_x_dict_conv = self.convolutional_pass(g2_x_dict,g2_edge_index_dict)

        g1_x_dict_conv_concat = torch.cat([g1_x_dict_conv[node_type] for node_type in g1_x_dict_conv.keys()], dim=0)
        g2_x_dict_conv_concat = torch.cat([g2_x_dict_conv[node_type] for node_type in g2_x_dict_conv.keys()], dim=0)


        g1_function_x=self.get_function_x(g1_function_dict,g1_x_dict_conv)
        g2_function_x=self.get_function_x(g2_function_dict,g2_x_dict_conv)


        hist=self.calculate_histogram(g1_function_x,g2_function_x) #(1,bins)
        #得到图向量
        pooled_features_1 = self.attention(g1_x_dict_conv_concat) #(heads*hidden_dim,1)
        pooled_features_2 = self.attention(g2_x_dict_conv_concat)

        scores = self.tensor_network(pooled_features_1, pooled_features_2) # (tensor-neurons,1)
        scores = torch.t(scores)

        scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores),inplace=True)
        score = torch.sigmoid(self.scoring_layer(scores)) #归一化到0-1 得到最终图相似度得分

        return score
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
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
        self.loss_fn = ContrastiveLoss(margin=0.01)

    def load_data(self):
        pass

    def initial(self):
        self.training_graphs = glob.glob(self.args.debug_training_graphs + "*.pth")
        testing_graphs = glob.glob(self.args.debug_testing_graphs + "*.pth")
        split_ratio = 0.2  # 20% 用作验证集
        num_validation = int(len(testing_graphs) * split_ratio)
        random.shuffle(testing_graphs)
        self.valid_graphs = testing_graphs[:num_validation]
        self.testing_graphs = testing_graphs[num_validation:]

        if len(self.training_graphs) % 2 != 0:
            self.training_graphs.pop()  
        if len(self.valid_graphs) % 2 != 0:
            self.valid_graphs.pop()  
        if len(self.testing_graphs) % 2 != 0:
            self.testing_graphs.pop()  
        


    def create_batches(self,graphs):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(graphs)
        batches = []
        for graph in range(0, len(graphs), self.args.batch_size):
            batches.append(graphs[graph:graph+self.args.batch_size])
        return batches

    def process_batch(self, batch):

        self.optimizer.zero_grad()  
        g_pairs = [(batch[i], batch[i + 1]) for i in range(0, len(batch), 2)]
        
        batch_g_sim=[]
        batch_labels=[]
        for g_pair in g_pairs:
            g1 = torch.load(g_pair[0])
            g2 = torch.load(g_pair[1])
            
            g1_x_dict = {
                'inst': g1['inst'].x,
                'data': g1['data'].x,
            }

            g1_edge_index_dict = {
            ('inst','control','inst'): g1['inst','control','inst'].edge_index,
                ('data','input','inst'): g1['data','input','inst'].edge_index,
                ('inst','output','data'): g1['inst','output','data'].edge_index,
                ('inst','call','inst'):g1['inst','call','inst'].edge_index
            }

            g2_x_dict = {
                'inst': g2['inst'].x,
                'data': g2['data'].x,
            }

            g2_edge_index_dict = {
            ('inst','control','inst'): g2['inst','control','inst'].edge_index,
                ('data','input','inst'): g2['data','input','inst'].edge_index,
                ('inst','output','data'): g2['inst','output','data'].edge_index,
                ('inst','call','inst'):g2['inst','call','inst'].edge_index
            }

            batch_labels.append(1 if g1.g_label == g2.g_label else 0)
            batch_g_sim.append(self.model((g1_x_dict,g2_x_dict),(g1_edge_index_dict,g2_edge_index_dict),(g1.function_dict,g2.function_dict)).reshape(-1))


        batch_g_sim = torch.cat(batch_g_sim)  
        batch_labels = torch.tensor(batch_labels, dtype=torch.float).cuda()
        #batch_labels = torch.tensor(batch_labels, dtype=torch.float)

        batch_avg_loss=self.loss_fn(batch_g_sim,batch_labels)
        batch_avg_loss.backward()
        self.optimizer.step()

        return batch_avg_loss.detach().item()

    def fit(self):
        """
        Fitting a model.
        """

        self.model = self.model.cuda()  
        #self.model=self.model
        
        print("\nModel training.\n")


        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        
        start_epoch=0

        if self.args.load_path:
            start_epoch=self.load()+1


        self.model.train()


        best_metric = float('inf')  # 用于存储最佳模型的验证误差
        epochs = trange(start_epoch,self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            batches = self.create_batches(self.training_graphs)
            epoch_loss_sum = 0 #统计这个epoch的总loss
            main_index = 0
            for batch_index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):

                batch_avg_loss = self.process_batch(batch) #得到batch的平均loss
                main_index = main_index + len(batch)
                epoch_loss_sum = epoch_loss_sum + batch_avg_loss*len(batch) 
                avg_loss = epoch_loss_sum/main_index
                epochs.set_description("Epoch %d (Loss=%g)" % (epoch,round(avg_loss, 10))) #得到当前的平均loss

            
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

        self.model.eval()
        self.ground_truth = []
        self.predition=[]

        if mode =='test':
            graphs=self.testing_graphs
            print("\nFinal evaluation on the test set.")
        elif mode == 'eval':        
            graphs=self.valid_graphs
        else:
            print('error mode .please use test or eval')
            import sys
            sys.exit(1)

        loss = 0 #统计总loss

        batches = self.create_batches(graphs)
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            batch_avg_loss = self.process_batch(batch) #得到batch的平均loss
            loss = loss + batch_avg_loss*len(batch) 

        return loss/len(graphs)  #返回验证集/测试集的平均loss

    def save(self,epoch,metric):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间格式: 年月日_时分秒

        # 组合文件名，包含时间、epoch 和 指标
        filename = f"{current_time}_epoch={epoch}_metric={metric:.5f}.pth"
        
        # 完整保存路径
        save_path = f"{self.args.save_dir}/{filename}"
        checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epoch': epoch
        }
        torch.save(checkpoint, save_path)
        print(f"Model and optimizer state saved to {save_path}.")

    def load(self):
        checkpoint = torch.load(self.args.load_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']

        return epoch

