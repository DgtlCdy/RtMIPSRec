# -*- coding: UTF-8 -*-
# @Author  : Chao Deng
# @Email   : dengch26@mail2.sysu.edu.cn


import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import utils

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel
from utils import layers

class RtMIPSRecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--time_size', type=int, default=256,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--time_max', type=int, default=256,
                            help='Max time intervals.')
        return parser        

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.time_size = args.time_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.max_time = args.time_max
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._base_define_params()
        self.apply(self.init_weights)
        self.R = 0
        self.gram_matrix = 0  # 把item相似矩阵放在base里面

        # 获得全部时间，并求得最大值、最小值、最小时间间隔，然后根据这些参数建模时间间隔embedding、确定索引方式
        time_seqs = []
        # 训练集中给出最小的时间戳
        for u, user_df in corpus.all_df.groupby('user_id'):
            time_seqs.extend(user_df['time'].values.tolist())
        # 测试集中给出最大的时间戳
        for u, user_df in corpus.data_df['test'].groupby('user_id'):
            time_seqs.extend(user_df['time'].values.tolist())
        time_seqs = sorted(set([int(_) for _ in time_seqs]))
        self.min_timestamp = time_seqs[0]
        self.max_timestamp = time_seqs[-1]
        self.min_interval = 0xFFFFFFFF
        for idx in range(len(time_seqs) - 1):
            self.min_interval = min(self.min_interval, time_seqs[idx+1] - time_seqs[idx])
        if self.min_interval == 0:
            self.min_interval = 1
        self.min_timestamp_converted = 0
        self.max_timestamp_converted = (self.max_timestamp - self.min_timestamp) / self.min_interval


    def _base_define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.t_embeddings_gm = nn.Embedding(self.max_time + 2, self.emb_size)
        self.t_embeddings_gp_k = nn.Embedding(self.max_time + 2, self.emb_size)
        self.t_embeddings_gp_v = nn.Embedding(self.max_time + 2, self.emb_size)
        # self.t_embeddings_sa = nn.Embedding(self.max_time + 2, self.emb_size)
        # self.t_embeddings_ffn = nn.Embedding(self.max_time + 2, self.emb_size)

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer_Rt(d_model=self.emb_size, d_ff=self.emb_size, d_t=self.time_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        t_history = feed_dict['history_times']  # [batch_size, history_max]
        t_target = feed_dict['target_time']  # item_id对应的time，包括训练、验证、测试的三个时间，规模与i_ids一致
        lengths = feed_dict['lengths']  # [batch_size] # 每一个用户序列的长度，取值1-20

        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()

        # 兴趣编码
        interests_sim = self.gram_matrix[history]
        interests_input = interests_sim @ self.i_embeddings.weight
        his_vectors = interests_input

        # 位置编码
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)

        # 获取每一个交互离最新交互的时间间隔current_interval
        t_history = t_history
        t_history = (t_history - self.min_timestamp).relu()
        t_history = t_history / self.min_interval
        max_values, _ = torch.max(t_history, dim=1)
        realtime = (t_target - self.min_timestamp) / self.min_interval
        # current_interval = max_values.unsqueeze(-1).expand_as(t_history) - t_history
        current_interval = realtime.unsqueeze(-1).expand_as(t_history) - t_history

        # 将时间间隔转化为时间Embedding，并索引对应的时间Embedding
        convert_log_a = torch.pow(torch.tensor(self.max_timestamp_converted), torch.tensor(1. / self.max_time))
        idx = (torch.log(current_interval + 1) / torch.log(convert_log_a)).int()
        t_ebds_m = self.t_embeddings_gm(idx) # 单调部分完成，但还没有卷积的部分


        # 叠加得到混合兴趣Embedding
        his_vectors = his_vectors + pos_vectors + t_ebds_m

        # 进行自注意编码
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int32)) # 只取下三角的矩阵，表示seq的邻接关系
        attn_mask = torch.from_numpy(causality_mask).to(torch.device('cuda'))
        attn_mask_full = torch.ones_like(attn_mask)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors, weight_t = block(his_vectors, attn_mask_full) # transformer的输出维度和输入维度是一样的
            # his_vectors = block(his_vectors, attn_mask) # transformer的输出维度和输入维度是一样的
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # 针对获取的时间取旋转的权重
        # 其实只需要知道最短时间间隔，也就是时间单位即可，amazon的数据集，最小时间间隔是86400秒，可以先用直接的时间来估计
        # 现在已知current_interval，即每个兴趣的相对时间间隔，而且是调整好的，现在要求出对应的每个权重
        # 假设周期最长为10年
        # 定义第一个元素、最后一个元素和长度
        # first_element = 3600 * 24
        # last_element = 3600 * 24 * 365
        if self.min_interval == 86400:
            unit_oneday = 1
        elif self.min_interval == 1:
            unit_oneday = 86400
        else:
            unit_oneday = 86400 / self.min_interval
        first_element = unit_oneday * 1
        last_element = unit_oneday * 10 * 365
        length = self.time_size

        # 计算公比
        # ratio = (last_element / first_element) ** (1 / (length - 1))
        # period = first_element * (ratio ** torch.arange(length)).float()
        # omega = (2 * torch.pi / period).to(self.device)
        # omega = (period / (2 * torch.pi) / last_element).to(self.device)
        # 计算公差
        ratio = (last_element - first_element) / length
        period = (first_element + (ratio * torch.arange(length))).float().to(self.device)
        omega = (2 * torch.pi / period).to(self.device)
        # omega = (period / (2 * torch.pi) / last_element).to(self.device)

        time_attenuation = period[None, None, :] / (period[None, None, :] + 0.01 * current_interval[:, :, None])


        # weight_t_added = weight_t * ((torch.cos(t_history[:, :, None] * omega[None, None, :]) + 1) / 2)
        # weight_t_added = weight_t * ((torch.cos(current_interval[:, :, None] * omega[None, None, :]) + 1) / 2)
        weight_t_added = weight_t * time_attenuation * ((torch.cos(current_interval[:, :, None] * omega[None, None, :]) + 1) / 2)

        weight_t_added = weight_t_added.sum(-1)
        his_vectors = his_vectors * weight_t_added[:, :, None]
        his_vectors = his_vectors * valid_his[:, :, None].float()


        # 只取最后一个item的embedding作为本次训练的预测embedding
        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :] # 为什么不直接用冒号？
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding

        i_vectors = self.i_embeddings(i_ids) # 获取阳性item和阴性item的embedding

        # 输出侧
        # 方法0：对最后一个输出embedding求内积
        # prediction = (his_vector[:, None, :] * i_vectors).sum(-1) # 获取和阳性item、阴性item的内积，前者越大越好后者越小越好
        # 方法1：把所有的vectors放一起求内积均值
        prediction = (his_vectors[:, None, :, :] * i_vectors[:, :, None, :])
        prediction = prediction.sum(-1).sum(-1)
        prediction = prediction[:, :] / lengths[:, None]


        u_v = his_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
        i_v = i_vectors

        # 返回一个字典。
        # prediction是预测的内积，训练时返回对两个指定item的内积，测试时返回100个item的id？
        # u_v是预测的embedding
        # i_v是阳性和阴性的embedding
        return {'prediction': prediction.view(batch_size, -1), 'kl': 0, 'u_v': u_v, 'i_v':i_v}


class RtMIPSRec(SequentialModel, RtMIPSRecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = RtMIPSRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def get_gram_matrix(self, dataset):
        R = torch.zeros(self.user_num, self.item_num)
        for (user_index, item_index) in zip(dataset.data['user_id'], dataset.data['item_id']):
            R[user_index, item_index] = 1
        self.R = R.to(self.device)

        row_sum = np.array(R.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten() #根号度分之一
        d_inv[np.isposinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv) # 对角的度矩阵
        norm_mat = d_mat.dot(R)
        col_sum = np.array(R.sum(axis=0))
        d_inv = np.power(col_sum, -0.5).flatten()
        d_inv[np.isposinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_mat = norm_mat.dot(d_mat.toarray()).astype(np.float32)
        gram_matrix = norm_mat.T.dot(norm_mat)
        gram_matrix =  torch.Tensor(gram_matrix).to(self.device)

        # 取高阶相似度
        item_embedding_r2 = gram_matrix @ self.R.T
        gram_matrix_r2 = item_embedding_r2 @ item_embedding_r2.T
        gram_matrix_r2 =  torch.nn.functional.normalize(gram_matrix_r2)
        gram_matrix_r2 = gram_matrix_r2 / gram_matrix_r2.mean() * gram_matrix.mean()
        gram_matrix = gram_matrix * 0.8 + gram_matrix_r2 * 0.2

        # 取top10%的相似度
        top = int(self.item_num * 0.1)
        indices = torch.topk(gram_matrix, top, dim=1).indices
        gram_matrix_topk = torch.zeros_like(gram_matrix)
        gram_matrix_topk.scatter_(1, indices, gram_matrix.gather(1, indices))
        gram_matrix_topk = torch.nn.functional.normalize(gram_matrix_topk, p=2)
        self.gram_matrix = gram_matrix_topk


    def forward(self, feed_dict):
        out_dict = RtMIPSRecBase.forward(self, feed_dict)
        # return {'prediction': out_dict['prediction']}
        return {'prediction': out_dict['prediction'], 'kl': out_dict['kl']}


class RtMIPSRecImpression(ImpressionSeqModel, RtMIPSRecBase):
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = RtMIPSRecBase.parse_model_args(parser)
        return ImpressionSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        ImpressionSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return RtMIPSRecBase.forward(self, feed_dict)