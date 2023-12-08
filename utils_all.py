import numpy as np
import pandas as pd
import logging
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.neighbors import kneighbors_graph
import torch.nn as nn
from collections import deque
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Blast import NCBIXML
#from Bio.Blast import NCBIStandalone
from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist
from Bio.Align import substitution_matrices
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse.linalg import eigs
from model.node_embed import Tree2Vec


def reshape(values):
    #values = np.hstack(values).reshape(len(values), len(values[0]))
    values = np.hstack(values).reshape(len(values), len(values[0]))
    return values

def get_values(data_frame,MAXLEN):
    print((data_frame['labels'].values.shape))
    labels = reshape(data_frame['labels'].values)
    # arg_labels=np.argmax(labels,axis=1)
    # arg_labels=np.reshape(arg_labels,newshape=(-1,1))
    # ngrams = sequence.pad_sequences(
    #     data_frame['ngrams'].values, maxlen=MAXLEN)### 进行0填充
    # 将序列 padded到最大长度MAXLEN
    # padded_ngrams = pad_sequence(data_frame['ngrams'].values, batch_first=True, padding_value=0)
    # ngrams = padded_ngrams[:, :MAXLEN]
    #ngrams = torch.nn.ConstantPad2d(padding=MAXLEN, value=0)
    #ngrams=np.pad(data_frame['ngrams'].values, ((0, 0), (0, MAXLEN-)), 'constant', constant_values=(0, 0))
    ngrams=data_frame['ngrams'].values
    ngrams = np.array([
        np.pad(ngrams[i], (0, MAXLEN- len(ngrams[i])), 'constant') for i in range(len(ngrams))
    ])
    ngrams = reshape(ngrams)
    rep = reshape(data_frame['embeddings'].values)
    # data = (ngrams, rep)
    ####在这里进行拼接
    data = np.concatenate((ngrams, rep), axis=-1)
    return data, labels

def get_resample_values(data_frame,MAXLEN):
    print((data_frame['labels'].values.shape))
    labels = reshape(data_frame['labels'].values)
    # ngrams = sequence.pad_sequences(
    #     data_frame['ngrams'].values, maxlen=MAXLEN)### 进行0填充
    # 将序列 padded到最大长度MAXLEN
    # padded_ngrams = pad_sequence(data_frame['ngrams'].values, batch_first=True, padding_value=0)
    # ngrams = padded_ngrams[:, :MAXLEN]
    #ngrams = torch.nn.ConstantPad2d(padding=MAXLEN, value=0)
    #ngrams=np.pad(data_frame['ngrams'].values, ((0, 0), (0, MAXLEN-)), 'constant', constant_values=(0, 0))
    ngrams=data_frame['ngrams'].values
    ngrams = np.array([
        np.pad(ngrams[i], (0, MAXLEN- len(ngrams[i])), 'constant') for i in range(len(ngrams))
    ])
    ngrams = reshape(ngrams)
    rep = reshape(data_frame['embeddings'].values)
    gos = data_frame["gos"].values
    gos = np.reshape(gos,(-1,1))
    # data = (ngrams, rep)
    ####在这里进行拼接
    data = np.concatenate((ngrams, rep,gos), axis=-1)
    ##过采样
    ros = RandomOverSampler(random_state=0)

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)  # 多标签转化为多分类编码
    data, labels = ros.fit_resample(data, labels)

    labels = mlb.inverse_transform(labels)  # 转回多标签格式

    return data[:,:-1],data[:,-1], labels


def load_data(DATA_ROOT,FUNCTION,ORG,MAXLEN):

    # df = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '-nomissing.pkl')
    df = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    n = len(df)
    index = df.index.values
    valid_n = int(n * 0.8)
    train_df = df.loc[index[:valid_n]]
    valid_df = df.loc[index[valid_n:]]
    #test_df = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '-nomissing.pkl')
    test_df = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    if ORG is not None:
        logging.info('Unfiltered test size: %d' % len(test_df))
        test_df = test_df[test_df['orgs'] == ORG]
        logging.info('Filtered test size: %d' % len(test_df))

    # Filter by type
    # org_df = pd.read_pickle('data/prokaryotes.pkl')
    # orgs = org_df['orgs']
    # test_df = test_df[test_df['orgs'].isin(orgs)]

    train = get_values(train_df,MAXLEN)
    valid = get_values(valid_df,MAXLEN)
    test = get_values(test_df,MAXLEN)
    return train, valid, test, train_df, valid_df, test_df

##### numpy数据标准化
def standardization(data):
    mu = np.mean(data, axis=0)# (1256,)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def load_data_mask(DATA_ROOT,FUNCTION,ORG,MAXLEN):
    ###num_train = 1000
    #df = pd.read_pickle(DATA_ROOT +FUNCTION + '.pkl')
    df1 = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    df2 = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    df=pd.concat([df1,df2],axis=0)
    gos = df["gos"].values
    sequences = df["sequences"].values
    proteins = df["proteins"].values
    seed=1
    n = len(df)
    index = df.index.values
    np.random.seed(1)
    shuffled_idx = np.random.permutation(np.array(range(n)))  # 已经被随机打乱
    train_idx = shuffled_idx[:int(0.90 * n)]
    val_idx = shuffled_idx[int(0.9 * n): int(0.901 * n)]
    test_idx = shuffled_idx[int(0.90 * n):]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = 1
    val_mask[val_idx]=1
    test_mask[test_idx]=1

    train_df = df.loc[train_idx]
    valid_df = df.loc[val_idx]
    test_df= df.loc[test_idx]
    if ORG is not None:
        logging.info('Unfiltered test size: %d' % len(test_df))
        test_df = test_df[test_df['orgs'] == ORG]
        logging.info('Filtered test size: %d' % len(test_df))
    all_values = get_values(df,MAXLEN)
    ## 数据预处理 ：在网络前进行处理
    #all_values = (standardization(all_values[0]),all_values[1])
    return all_values,train_mask,val_mask,test_mask,shuffled_idx,gos,sequences,proteins

def load_data_org(DATA_ROOT,FUNCTION,ORG,MAXLEN):
    ###num_train = 1000
    #df = pd.read_pickle(DATA_ROOT +FUNCTION + '.pkl')
    df1 = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    df2 = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    df=pd.concat([df1,df2],axis=0)
    gos = df["gos"].values
    sequences = df["sequences"].values
    proteins = df["proteins"].values
    seed=1
    n = len(df)
    index = df.index.values
    np.random.seed(1)
    shuffled_idx = np.random.permutation(np.array(range(n)))  # 已经被随机打乱
    train_idx = shuffled_idx[:int(0.90 * n)]
    val_idx = shuffled_idx[int(0.9 * n): int(0.901 * n)]
    test_idx = shuffled_idx[int(0.90 * n):]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = 1
    val_mask[val_idx]=1
    #test_mask[test_idx]=1
    print("the chosen org is ", ORG)

    train_df = df.loc[train_idx]
    valid_df = df.loc[val_idx]
    test_df= df.loc[test_idx]
    if ORG is not None:
        logging.info('Unfiltered test size: %d' % len(test_idx))
        test_df = test_df[test_df['orgs'] == ORG]

        orgs_index = np.where(df["orgs"].values==ORG)[0]
        sname_index = np.intersect1d(orgs_index, test_idx)
        test_mask[sname_index] = 1
        logging.info('Filtered test size: %d' % len(sname_index))
    all_values = get_values(df,MAXLEN)
    ## 数据预处理 ：在网络前进行处理
    #all_values = (standardization(all_values[0]),all_values[1])
    return all_values,train_mask,val_mask,test_mask,shuffled_idx,gos,sequences,proteins

def load_resample_data(DATA_ROOT,FUNCTION,ORG,MAXLEN):
    ###num_train = 1000
    #df = pd.read_pickle(DATA_ROOT +FUNCTION + '.pkl')
    df1 = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    df2 = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    df = pd.concat([df1,df2],axis=0)
    gos = df["gos"].values
    seed=1

    all_values=get_resample_values(df,MAXLEN)
    gos = all_values[1]
    all_values = (all_values[0],all_values[-1])

    n = all_values[0].shape[0]
    index = df.index.values
    shuffled_idx = np.random.permutation(np.array(range(n)))  # 已经被随机打乱
    train_idx = shuffled_idx[:int(0.9 * n)]
    val_idx = shuffled_idx[int(0.9 * n): int(0.901 * n)]
    test_idx = shuffled_idx[int(0.90 * n):]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1

    return all_values,train_mask,val_mask,test_mask,shuffled_idx,gos

########## 氨基酸序列转换为索引 #####
def AA_sequences(df):
    # df1 = pd.read_pickle(args["DATA_ROOT"] + 'train' + '-' + args["function"] + '.pkl')
    # df2 = pd.read_pickle(args["DATA_ROOT"] + 'test' + '-' + args["function"] + '.pkl')
    # df = pd.concat([df1,df2],axis=0)
    #### 20种氨基酸
    aa_vocab = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    sequences = df["sequences"].values
    max_len = 0
    aa_indexed = [[aa_vocab.index(aa) for aa in seq] for seq in sequences]# 0-19表示氨基酸
    for i in range(sequences.shape[0]):
        if max_len < len(aa_indexed[i]):
            max_len = len(aa_indexed[i])
    max_len = max_len + 1
    print(max_len)
    # sequences = np.array([
    #         np.pad(aa_indexed[i], (0, max_len- len(aa_indexed[i])), 'constant',constant_values=(-1,-1)) for i in range(len(aa_indexed))
    #     ])##填充20? 填充-1?
    sequences = np.array([
        np.pad(aa_indexed[i], (0, max_len - len(aa_indexed[i])), 'constant',constant_values=20) for i in
        range(len(aa_indexed))
    ])  ##填充20? 填充-1?

    return sequences,max_len

def load_sequences(DATA_ROOT,FUNCTION,ORG,MAXLEN):
    ###num_train = 1000
    #df = pd.read_pickle(DATA_ROOT +FUNCTION + '.pkl')
    df1 = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    df2 = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    df = pd.concat([df1,df2],axis=0)
    gos = df["gos"].values
    sequences = df["sequences"].values
    proteins = df["proteins"].values
    labels = reshape(df['labels'].values)
    seed=1
    n = len(df)
    index = df.index.values
    np.random.seed(1)
    shuffled_idx = np.random.permutation(np.array(range(n)))  # 已经被随机打乱
    train_idx = shuffled_idx[:int(0.90 * n)]
    val_idx = shuffled_idx[int(0.9 * n): int(0.901 * n)]
    test_idx = shuffled_idx[int(0.90 * n):]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = 1
    val_mask[val_idx]=1
    test_mask[test_idx]=1

    train_df = df.loc[train_idx]
    valid_df = df.loc[val_idx]
    test_df= df.loc[test_idx]
    if ORG is not None:
        logging.info('Unfiltered test size: %d' % len(test_df))
        test_df = test_df[test_df['orgs'] == ORG]
        logging.info('Filtered test size: %d' % len(test_df))

    sequences,max_sequences = AA_sequences(df)
    print(sequences.shape)
    print(np.unique(sequences))
    all_values = (sequences,labels)
    ## 数据预处理 ：在网络前进行处理
    #all_values = (standardization(all_values[0]),all_values[1])
    return all_values,train_mask,val_mask,test_mask,shuffled_idx,gos,sequences,proteins,max_sequences

def get_Uniprot_ID(DATA_ROOT,FUNCTION,ORG,MAXLEN):
    ###num_train = 1000
    #df = pd.read_pickle(DATA_ROOT +FUNCTION + '.pkl')
    df1 = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    df2 = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    df = pd.concat([df1,df2],axis=0)
    gos = df["gos"].values
    sequences = df["sequences"].values
    proteins = df["proteins"].values
    labels = reshape(df['labels'].values)
    Uniprot_ID = df["accessions"].values

    return Uniprot_ID

def get_parents(go, go_id):
    go_set = set()
    for parent_id in go[go_id]['is_a']:
        if parent_id in go:
            go_set.add(parent_id)
    return go_set


def get_gene_ontology(filename):
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in list(go.keys()):
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    return go


def get_node_name(go_id, node_names,unique=False):
    name = go_id.split(':')[1]
    if not unique:
        return name
    if name not in node_names:
        node_names.add(name)
        return name
    i = 1
    while (name + '_' + str(i)) in node_names:
        i += 1
    name = name + '_' + str(i)
    node_names.add(name)
    return name

def get_function_node(name, inputs):

    output_name = name + '_out'
    len_1=inputs.shape[1]
    #net = Dense(units=256, name=name, activation='relu',input_shape=(32,))(inputs)#TensorShape([Dimension(None), Dimension(None)])-->Out[1]: TensorShape([Dimension(None), Dimension(256)])
    net = nn.Sequential(
        nn.Linear(len_1, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    output = net(inputs)
    return net, output


def BLAST(query, subject_seqs):
    # # 用makeblastdb创建蛋白数据库
    # makedb_cline = NcbiblastpCommandline(subject_seqs, dbtype="prot")
    # makedb_cline()

    # BLAST比对
    blastp_cline = NcbiblastpCommandline(query=query,
                                         db=subject_seqs,
                                         evalue=0.001,
                                         outfmt=5)
    #blast_records = NcbiblastpCommandline(cmd=blastp_cline)()
    # BLAST比对
    result_handle = NcbiblastpCommandline(cmd=blastp_cline)()

    # 解析比对结果
    blast_records = NCBIXML.parse(result_handle)
    # # BLAST比对
    # blastn_cline = NCBIStandalone.BlastnCommandline(query=query,
    #                                                 subject=subject_seqs,
    #                                                 outfmt=5)
    # blast_records = NCBIStandalone.BlastnCommandline(cmd=blastn_cline)[0]
    return blast_records

# def normalize_adjacency_matrix(adjacency_matrix):
#     # 添加自循环：函数会添加自循环，即将邻接矩阵的对角线元素设置为1
#     adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0])
#
#     # 计算度矩阵
#     degree_matrix = np.sum(adjacency_matrix, axis=1)
#
#     # 计算度矩阵的逆矩阵
#     degree_matrix_inverse = np.diag(1 / degree_matrix)
#     print(degree_matrix_inverse.shape)
#     # 归一化邻接矩阵
#     normalized_adjacency_matrix = degree_matrix_inverse.dot(adjacency_matrix).dot(degree_matrix_inverse)
#
#     return normalized_adjacency_matrix
def normalize_adjacency_matrix(adj):
    # 加入自循环
    adj = adj + np.eye(adj.shape[0])

    # 度矩阵
    d = np.array(adj.sum(1))

    # 归一化
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

    return norm_adj

def compute_adj(x , n_class):
    ## 方法1：构建k近邻图
    adj = kneighbors_graph(x, n_neighbors=n_class, mode='connectivity')#初步分类
    # # 方法2：计算所有蛋白质序列之间的相似性打分,静态构建邻接矩阵
    # scores = np.pairwise_sequence_similarity(x)
    #
    # # 根据阈值保留高相似性边
    # threshold  = 0.5
    # adj = np.where(scores > threshold, scores, 0)

    # # 方法2：余弦相似度 ######
    # n = x.shape[0]
    # adj = np.zeros((n, n))
    #
    # for i in range(n):
    #     for j in range(n):
    #         if i != j:
    #             sim = np.dot(x[i,:], x[j,:]) / (np.linalg.norm(x[i,:]) * np.linalg.norm(x[j,:]))
    #             adj [i, j] = sim
    #### 方法三：BLAST方法 ####
    ### 实现一个进行BLAST比对的函数,输入为查询序列和目标序列数据库,输出比对的E值或分数。
    # # # # 建立数据库和查询序列
    # subject_seqs = x # (31530, 1256)
    # # 构建邻接矩阵
    # n = x.shape[0]
    # adj = np.zeros((n, n))
    #
    # for i, query in enumerate(x):
    #     blast_records = BLAST(query, subject_seqs)
    #     for alignment in blast_records:
    #         j = alignment.title
    #         evalue = -np.log(alignment.hsps[0].expect)
    #         adj[i, j] = evalue
    #adjs = torch.where(adj > 0.5, adj, 1)

    ############### 方法四：Needleman-Wunsch算法计算两个蛋白质序列相似度
    # # 定义评分函数
    # def nw_score(pair):
    #     if pair[0] == pair[1]:
    #         return 1
    #     else:
    #         return 0
    # # 初始化邻接矩阵
    # n = x.shape[0]
    # adj = np.zeros((n, n))
    # # 获取Blosum62评分矩阵
    # # #blosum62 = matlist.blosum62
    # # blosum62 = np.array(substitution_matrices.load("BLOSUM62").values())
    # # blosum62 = np.reshape(blosum62,newshape=(-1,1)) #### (576, 1)
    # # 计算序列相似度
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         seq1, seq2 = x[i], x[j]
    #         # Needleman-Wunsch算法
    #         align = pairwise2.align.globalms(seq1, seq2, 1, -1, -1, -1)
    #         optimal = align[0]
    #         # 计算序列i和j的相似度
    #         matches = 0
    #         for a, b in zip(optimal[0], optimal[1]):
    #             if a == b:
    #                 matches += 1
    #         seq_sim = matches / len(optimal[0])
    #         adj[i, j] = seq_sim
    #         adj[j, i] = seq_sim
    return adj

### 构建 go-go之间的邻接矩阵
def build_GoAdj(args):
    adj_mx = np.zeros(shape=(args["nb_classes"],args["nb_classes"]))
    triples = []
    ######### 读取关系 triplet ########
    for node_id in args["functions"]:
        childs = set(args["go"][node_id]['children']).intersection(args["func_set"])
        if len(childs) > 0:
            for ch_id in childs:
                triples.append((node_id, 'is_a', ch_id))
    Functions = args["functions"]
    ########## 构建邻接矩阵 ########
    for s, r, o in triples:
        s_index = np.where(Functions==s)[0][0]
        o_index = np.where(Functions == o)[0][0]
        adj_mx[s_index, o_index] = 1

    return normalize_adjacency_matrix(adj_mx)

### 构建 go-go之间的邻接矩阵
def build_GoStructure(args):
    adj_mx = np.zeros(shape=(args["nb_classes"],args["nb_classes"]))
    triples = []
    ######### 读取关系 triplet ########
    for node_id in args["functions"]:
        childs = set(args["go"][node_id]['children']).intersection(args["func_set"])
        if len(childs) > 0:
            for ch_id in childs:
                triples.append((node_id, 'is_a', ch_id))
    Functions = args["functions"]
    ########## 构建邻接矩阵 ########
    for s, r, o in triples:
        s_index = np.where(Functions==s)[0][0]
        o_index = np.where(Functions == o)[0][0]
        adj_mx[s_index, o_index] = 1

    function = args["function"]
    best_model = Tree2Vec(input_dim=adj_mx.shape[0])
    best_model.load_state_dict(torch.load(f"experiments\\best_embed_{function}.pth"))
    node_features = best_model(adj_mx,args)
    node_features = node_features.detach().numpy()
    return normalize_adjacency_matrix(adj_mx) , node_features

def compute_similarity(seq1, seq2):
    # 在这里实现计算序列相似度的方法
    # 这里使用简单的编辑距离作为相似度度量，你可以根据需要选择其他方法
    # 返回值为两个序列的相似度得分
    return 1 - (np.edit_distance(seq1, seq2) / max(len(seq1), len(seq2)))

def construct_adjacency_matrix(sequences):
    n = len(sequences)
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            similarity_score = compute_similarity(sequences[i], sequences[j])
            adjacency_matrix[i, j] = similarity_score
            adjacency_matrix[j, i] = similarity_score

    return adjacency_matrix


##把邻接矩阵分成batch
def batch_adj(adj,batch_size):# Storage存储分批后的邻接矩阵
    adj_batches = []
    # 计算可以分成的批数
    num_batches = adj.shape[0] // batch_size
    # if adj.shape[0]%batch_size==0:
    #     num_batches = num_batches
    # else:
    #     num_batches = num_batches+1
    # 逐批分割
    for i in range(num_batches):
        # 计算当前批的索引范围
        start = i * batch_size
        end = start + batch_size
        # if end > adj.shape[0]-1:
        #     end = adj.shape[0]-1
        # 取出当前批的邻接矩阵,归一化
        adj_batch = normalize_adjacency_matrix(adj[start:end, start:end])
        # 放入Storage
        adj_batches.append(adj_batch)
    ###### 最后一个batch ###
    # 将Storage转换为Tensor
    #adj_batches = torch.stack(adj_batches)#torch.Size([num_batches, batch_size, batch_size])
    adj_batches = [torch.from_numpy(b.A) for b in adj_batches]
    adj_batches = torch.stack(adj_batches)
    return adj_batches

def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
    return go_set


def compute_performance(preds, labels, gos,all_functions,GO_ID,func_set,go):
    data = preds
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    ##等一下加上去
    preds = (data - min_vals) / (max_vals - min_vals)

    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    predictions_max = 0
    recall_list, predictions_list=[],[]
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            all_gos = set()
            #### 根据本体层级关系,调整FN的计算,防止预测漏报父类的情况。
            for go_id in gos[i]:
                if go_id in all_functions:
                    all_gos |= get_anchestors(go, go_id)
            all_gos.discard(GO_ID)
            all_gos -= func_set
            fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if total<=0:
            continue
        r /= total
        if p_total<=0:
            continue
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            predictions_list.append(p)
            recall_list.append(r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max, recall_list,predictions_list

def compute_roc(preds, labels,args):
    # Compute ROC curve and ROC area for each class
    labels = np.reshape(labels,newshape=(-1 , 1))
    preds = np.reshape(preds, newshape=(-1, 1))
    fpr, tpr, _ = roc_curve(y_true=labels.astype('int'), y_score=preds)
    #roc_curve 函数不支持多类别格式的标签和预测结果。ROC曲线通常用于二分类问题，其中只有两个类别。
    roc_auc = auc(fpr, tpr)
    return roc_auc

import torch


def compute_performance2(y_pred,y_true,args):
    # # 假设以下变量已定义
    # y_pred = np.argmax(y_pred, axis=1)
    #
    # # 计算混淆矩阵
    # confusion_matrix = np.zeros((args["nb_classes"], args["nb_classes"]))
    # for i in range(len(y_true)):
    #     confusion_matrix[int(y_true[i]), int(y_pred[i])] = confusion_matrix[int(y_true[i]), int(y_pred[i])]+ 1
    #

    # # 计算TP、FP、FN、TN
    # # 通过计算混淆矩阵，可以得到每个类别的TP、FP、FN和TN。然后，根据这些值计算精确率、召回率和F1值。
    # TP = np.diag(confusion_matrix)
    # FP = np.sum(confusion_matrix, axis=0) - TP
    # FN = np.sum(confusion_matrix, axis=1) - TP
    # TN = np.sum(confusion_matrix) - (TP + FP + FN)
    #
    # # 计算精确率、召回率和F1值
    # precision = np.mean(TP / (TP + FP))
    # recall = np.mean(TP / (TP + FN))
    # f1 = np.mean(2 * (precision * recall) / (precision + recall))
    # Flatten the labels and predictions
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    # Compute the confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    # Compute TP, FP, FN, TN
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # Compute precision, recall, and F1-score
    precision = precision_score(y_true_flat, y_pred_flat, average='macro')
    recall = recall_score(y_true_flat, y_pred_flat, average='macro')
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro')

    return TP.item(), FP.item(), FN.item(), TN.item(), precision.item(), recall.item(), f1.item()
    #return TP, FP, FN, TN, precision, recall, f1


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self,args):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = args["nb_classes"]
        self.weights = torch.ones(self.num_classes,dtype=torch.float32).to(args["device"])

    def forward(self, inputs, targets):
        #loss = nn.CrossEntropyLoss(weight=self.weights)(inputs.to(torch.float), targets.long())
        loss = nn.CrossEntropyLoss(weight=self.weights)(inputs.float(), targets.long())
        return loss

    ####### 设置类别权重
    def update_weights(self, class_samples):
        # 将样本数量为0的类别替换为一个较小的非零值
        #class_samples = torch.where(class_samples > 0, class_samples, torch.tensor(1e-6))
        class_samples = torch.where(class_samples > 10, class_samples, torch.tensor(10))
        #class_samples = torch.where(class_samples < 100, class_samples, torch.tensor(100))
        total_samples = torch.sum(class_samples)
        #class_weights = total_samples / class_samples#torch.Size([589])
        class_weights = class_samples / total_samples  # torch.Size([589])
        self.weights = class_weights / torch.sum(class_weights)

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, args):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.num_classes = args["nb_classes"]
        self.weights = torch.ones(self.num_classes, dtype=torch.float32).to(args["device"])

    def forward(self, inputs, targets):
        loss = nn.BCEWithLogitsLoss(weight=self.weights)(inputs.float(), targets)
        #loss = nn.BCELoss(weight=self.weights)(inputs, targets)
        return loss

    ####### 设置类别权重
    def update_weights(self, class_samples):
        # 将样本数量为0的类别替换为一个较小的非零值
        #class_samples = torch.where(class_samples > 0, class_samples, torch.tensor(1e-6))
        class_samples = torch.where(class_samples > 10, class_samples, torch.tensor(10))
        #class_samples = torch.where(class_samples < 500, class_samples, torch.tensor(500))
        total_samples = torch.sum(class_samples)
        #class_weights = total_samples / class_samples#torch.Size([589])
        class_weights = class_samples / total_samples  # torch.Size([589])
        self.weights = class_weights / torch.sum(class_weights)

def scaled_Laplacian(W,args):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: torch.tensor, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''
    W = W.detach().cpu().numpy()
    lambda_max = 0
    if len(W.shape)==2:
        assert W.shape[0] == W.shape[1]

        #D = torch.diag(torch.sum(W,dim=1))

        D = np.diag(np.sum(W, axis=1))

        L = D - W

        lambda_max = eigs(L, k=1, which='LR')[0].real
        lambda_max = (2 * L) / lambda_max - np.identity(W.shape[0])
        lambda_max = torch.tensor(lambda_max).to(torch.float64).to(args["device"])
    else:
        lambda_max = torch.zeros(size=(W.shape[0],W.shape[1],W.shape[2])).to(torch.float64).to(args["device"])
        for i in range(W.shape[0]):
            w = W[i,:,:]
            D = np.diag(np.sum(w, axis=1))
            L = D - w
            lambda_1 = eigs(L, k=1, which='LR')[0].real
            lambda_1 = (2 * L) / lambda_1 - np.identity(w.shape[0])
            lambda_1 = torch.tensor(lambda_1).to(torch.float64).to(args["device"])
            lambda_max[i,:,:] = lambda_1
    #lambda_max = torch.tensor(lambda_max).to(torch.float64).to(args["device"])
    return lambda_max

