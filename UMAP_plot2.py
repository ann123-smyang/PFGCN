#from torch_geometric.nn import GCNConv
from model.PFGCN import PFGCNgo,PGCN,FGCN
#from model.PFGCN import PFGCN
import time
import os
from utils_all import *
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
# 导入CSV安装包
import csv
import umap
from sklearn.manifold import TSNE

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
conjName=r"conj\PFGCN.json"
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
def read_config():
    with open(conjName,encoding='UTF-8') as json_file:
        config = json.load(json_file)
    config["org"]= None
    config["supports"] = None
    config["node_names"] = set()
    config["go"] = get_gene_ontology(config["go_path"])
    func_df = pd.read_pickle(config["DATA_ROOT"] + config["function"] + '.pkl')
    functions = func_df['functions'].values
    config["functions"] = functions
    config["func_set"] = set(functions)
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {
        'cc': CELLULAR_COMPONENT,
        'mf': MOLECULAR_FUNCTION,
        'bp': BIOLOGICAL_PROCESS}
    config["Go_id"] = FUNC_DICT[config["function"]]
    config["MAXLEN"] = config["MAXLEN"]
    return config

############ 1. 导入数据 ###########
def main():
    # set parameters:
    # parser=parse_args()
    # args = parser.parse_args()
    args = read_config()
    func_df = pd.read_pickle(args["DATA_ROOT"] + args["function"] +'.pkl')
    functions = func_df['functions'].values
    nb_classes = len(functions)
    args["nb_classes"]=nb_classes
    start_time = time.time()
    print("function:",args["function"])
    logging.info("Loading Data")
    # train, val, test, train_df, valid_df, test_df = load_data(DATA_ROOT=args.DATA_ROOT,FUNCTION=args.function,ORG=args.org,MAXLEN=args.MAXLEN)
    all_values,train_mask,val_mask,test_mask,shuffled_idx , gos, sequences,_ = load_data_mask(DATA_ROOT=args["DATA_ROOT"], FUNCTION=args["function"],
                                                              ORG=args["org"], MAXLEN=args["MAXLEN"])
    # all_values, train_mask, val_mask, test_mask, shuffled_idx, gos = load_resample_data(DATA_ROOT=args["DATA_ROOT"],
    #                                                                                    FUNCTION=args["function"],
    #                                                                                   ORG=args["org"],
    #                                                                                    MAXLEN=args["MAXLEN"])
    # value11 = pd.DataFrame(all_values[0],columns=['all_values1'])
    # value12 = pd.DataFrame(all_values[1], columns=['all_values2'])
    # value2 = pd.DataFrame(train_mask, columns=['train_mask'])
    # value3 = pd.DataFrame(val_mask, columns=['val_mask'])
    # value4 = pd.DataFrame(all_values, columns=['test_mask'])
    # value5 = pd.DataFrame(shuffled_idx, columns=['shuffled_idx'])
    # value6 = pd.DataFrame(gos, columns=['gos'])
    # value = pd.concat([value11,value12, value2, value3,value4,value5,value6])
    # dataname = 'samples'+str(args["function"])+'.pkl'
    # value.to_pickle(dataname)
    # value = pd.read_pickle(value)
    # all_values = value["all_values1"].values,value["all_values2"].values
    # train_mask, val_mask, test_mask, shuffled_idx, gos =value["train_mask"].values,value["val_mask"].values,value["test_mask"].values,value["shuffled_idx"].values,value["gos"].values

    #data_gos = all_values['gos'].values
    logging.info("Data loaded in %d sec" % (time.time() - start_time))

    ########################## 加载数据为batch ########################
    x = all_values[0]
    y = all_values[1]
    ### add
    args["functions"] = args["functions"][:y.shape[1]]
    args["nb_classes"] = y.shape[1]
    all_values = np.concatenate((x, y), axis=1).astype(float)
    ###下面加载不打乱数据的顺序
    dataset= TensorDataset(torch.tensor(all_values).to(args["device"]))# (31530, 1845)--> (428678, 1257)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"])
    data_train_mask = TensorDataset(train_mask.clone().detach().to(args["device"]))
    train_mask_loader = DataLoader(data_train_mask, batch_size=args["batch_size"])

    data_val_mask = TensorDataset(val_mask.clone().detach().to(args["device"]))
    val_mask_loader = DataLoader(data_val_mask, batch_size=args["batch_size"])

    data_test_mask = TensorDataset(test_mask.clone().detach().to(args["device"]))
    test_mask_loader = DataLoader(data_test_mask, batch_size=args["batch_size"])

    idex = np.array(range(x.shape[0]))
    idex = TensorDataset(torch.tensor(idex).to(args["device"]))
    idex_loader = DataLoader(idex, batch_size=args["batch_size"])

    ########################## 生成静态邻接矩阵 ########################
    ## static_adj=con_adj(all_values,n_class=args["nb_classes")
    logging.info("build adj %d sec" % (time.time() - start_time))
    static_adj = compute_adj(x=x, n_class=100)
    ##把邻接矩阵分成一个一个batch
    adj_batches = batch_adj(adj=static_adj, batch_size=args["batch_size"])
    go_adj , node_features = build_GoStructure(args)
    node_features = torch.tensor(node_features).to(args["device"])
    logging.info("finish building adj %d sec" % (time.time() - start_time))
    ########### 2. 3. 训练 ###########
    model = PFGCNgo(args).to(args["device"])
    #model = ProteInfer(args).to(args["device"])
    #model.cuda()
    #optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"],weight_decay=args["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    ######### 定义了新的loss函数 #########
    #criterion = WeightedCrossEntropyLoss(args) ## loss
    criterion = WeightedBCEWithLogitsLoss(args)  ## loss
    ######### 更新权重 #########
    # logging.info("update the weight of loss funtion %d sec" % (time.time() - start_time))
    # class_samples = torch.tensor(np.sum(y)).to(args["device"])#.to(torch.float64)
    # criterion.update_weights(class_samples) ##更新权重

    ##### 训练 #####
    logging.info("strat training %d sec" % (time.time() - start_time))
    min_loss = float('inf')
    best_epoch = 0
    y_train = y[train_mask.detach().cpu().numpy()]
    train_gos_all = gos[train_mask.detach().cpu().numpy()]
    args["epochs"]= 0
    for epoch in range(args["epochs"]):
        i = 0
        train_Loss = 0
        #Ap = 0
        all_out = torch.empty((0, args["nb_classes"])).to(args["device"])
        for batch,the_train_mask,idx in zip(dataloader, train_mask_loader,idex_loader):
            if i >= static_adj.shape[0]//args["batch_size"]:
                continue
            batch_x,batch_y = batch[0][:,:-1*args["nb_classes"]] ,batch[0][:,-1*args["nb_classes"]:]
            model.train()

            batchAdj = adj_batches[i,:,:].clone().detach().to(args["device"])
            args["supports"] = batchAdj
            optimizer.zero_grad()
            out = model(batch_x,node_features=node_features,go_adj=go_adj,batch_adj=batchAdj)#torch.Size([64, 589])
            out = out[the_train_mask[0], :]
            batch_y = batch_y[the_train_mask[0],:]

            loss = criterion(out, batch_y)  ##新loss函数
            #loss = F.cross_entropy(out, batch_y.long())
            #loss = F.nll_loss(out, batch_y.long())
            # train_gos = gos[idx[0].cpu()][train_mask[0].cpu()]
            # ap = accuracy_score(y_true=batch_y.detach().cpu().numpy(),
            #                     y_pred=np.reshape(np.argmax(out.detach().cpu().numpy(), axis=1),
            #                                                 newshape=(-1, 1)))
            #                                                  ,average="macro")
            # f_max, p_max, r_max, t_max, predictions_max = compute_performance(out.detach().cpu().numpy()
            #                                                             ,batch_y.detach().cpu().numpy()
            #                                                             ,gos=train_gos,all_functions=args["functions"]
            #                                                             ,GO_ID=args["Go_id"],func_set=args["func_set"],go=args["go"])
            # f_max_all = f_max + f_max_all
            # p_max_all = p_max_all+p_max
            # r_max_all = t_max_all +r_max
            # predictions_max_all= predictions_max_all + predictions_max
            # roc_auc = compute_roc(out.detach().cpu().numpy(),batch_y.detach().cpu().numpy())
            train_Loss = train_Loss + loss
            # roc_auc_all =roc_auc_all+roc_auc
            #Ap = Ap + ap
            all_out = torch.concat((all_out, out), dim=0)
            #loss = F.cross_entropy(out[train_mask[0], :], batch_y[train_mask[0]])
            loss.backward()
            optimizer.step()

            i=i+1

        #print("the loss and acc of epoch %d is %.3f and %.4f."% (epoch,Loss,Ap/i))
        print("the loss of epoch %d is %f." % (epoch, train_Loss))
        #print("'Fmax measure: \t %f %f %f %f %f'" % (f_max, p_max, r_max, t_max,roc_auc))
        # 如果当前F1得分更高,则保存模型
        if train_Loss < min_loss:
            min_loss = train_Loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"experiments\\PFGCN\\best_model_{best_epoch}.pth")
        # f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_out.detach().cpu().numpy()
        #                                                                   , y_train
        #                                                                   , gos=train_gos_all
        #                                                                   , all_functions=args["functions"]
        #                                                                   , GO_ID=args["Go_id"], func_set=args["func_set"]
        #                                                                   , go=args["go"])
        # roc_auc = compute_roc(all_out.detach().cpu().numpy(), y, args)  # torch.Size([3141, 589]),(31530, 1) 多维数据的降维函数
        # #print("the loss and acc of training is %.3f and %.4f." % (Loss, Ap / i))
        # print("the Fmax measure of f_max, p_max, r_max, t_max, roc_auc: \t %f %f %f %f %f." % (f_max, p_max, r_max, t_max, roc_auc))
    ###### 最好epoch的进行测试
    # 加载最佳模型
    best_epoch = 28
    print("the best epoch is",best_epoch)
    best_model = PFGCNgo(args).to(args["device"])
    best_model.load_state_dict(torch.load(f"experiments\\PFGCN\\best_model_{best_epoch}.pth"))

    ########################## 在测试集上进行预测 ##########################
    i = 0
    test_Loss = 0
    #Ap = 0
    #best_model= best_model
    #all_out = torch.empty((0,args["nb_classes"])).to(args["device"])
    all_out = np.empty((0, 512))#.to(args["device"]) ,torch.Size([64*nbclass, new_feature])
    # test_y = np.empty((0,args["nb_classes"]))#.to(args["device"])#torch.Size([1, 589])
    #all_out = np.empty((0,args["nb_classes"], 32))  # .to(args["device"]) ,torch.Size([64*nbclass, new_feature])
    test_y = np.empty((0, args["nb_classes"]))  # .to(args["device"])#torch.Size([1, 589])
    test_gos_all = gos[test_mask.detach().cpu().numpy()]
    best_model.eval()
    for batch, the_test_mask , idx in zip(dataloader, test_mask_loader,idex_loader):
        if i >= static_adj.shape[0] // args["batch_size"]:
            break
        batch_x, batch_y = batch[0][:, :-1 * args["nb_classes"]], batch[0][:, -1 * args["nb_classes"]:]
        if batch_x.shape[0]!= args["batch_size"]:
            continue
        batchAdj = adj_batches[i, :, :].clone().detach().to(args["device"])
        #out = best_model(batch_x,node_features=node_features, go_adj=go_adj,batch_adj=batchAdj)  # torch.Size([64, 589])
        #loss = criterion(out, batch_y)  ##新loss函数
        #[64, nbclass, new_feature] ------>
        out = best_model.getFeature(batch_x, node_features=node_features, go_adj=go_adj, batch_adj=batchAdj)
        out = out[the_test_mask[0], :].detach().cpu().numpy()
        batch_y = batch_y[the_test_mask[0],:].detach().cpu().numpy()
        #loss = F.cross_entropy(out, batch_y.long())
        #loss = F.nll_loss(out, batch_y.long())
        test_gos = gos[idx[0].cpu()][test_mask[0].cpu()]

        if batch_y.shape[0]==0:
            continue
        #Ap = Ap + ap
        all_out = np.concatenate((all_out, out), axis=0)
        test_y = np.concatenate((test_y,batch_y), axis=0)
        # loss.backward()
        # optimizer.step()
        torch.cuda.empty_cache()
        i = i + 1

    ############################### 使用 UMAP 或者t-SNE 方法将数据嵌入到低维空间 ###############################
    #umap_embedding = umap.UMAP(n_neighbors=10, min_dist=0.2, n_components=2, random_state=0)
    n_components = 2
    umap_embedding = umap.UMAP(n_neighbors=20, min_dist=0.2, n_components=n_components,n_epochs=10)
    # embedding = umap_embedding.fit_transform(all_out)  # (100, 2),输入特征X，输出降维后的特征embedding

    # 使用 t-SNE 进行降维，降到2维
    # tsne = TSNE(n_components=2, random_state=42)
    # embedding = tsne.fit_transform(all_out)

    # 绘制嵌入空间的可视化图像
    # 获取不同类别的索引
    #################################### 为每个类别设置不同颜色，并绘制散点图 ##################
    ###### 找出最多人的label

    sumLabel = np.sum(test_y, axis=0)
    maxlabel_1 = np.argmax(sumLabel)
    sumLabel[maxlabel_1] = 0
    maxlabel_2 = np.argmax(sumLabel)
    sumLabel[maxlabel_2] = 0
    maxlabel_3 = np.argmax(sumLabel)
    sumLabel[maxlabel_3] = 0

    MaxLabel = [maxlabel_1, maxlabel_2, maxlabel_3]  # 最多的标签
    print(functions[maxlabel_1], functions[maxlabel_2], functions[maxlabel_3])
    subset_label = [functions[maxlabel_1], functions[maxlabel_2], functions[maxlabel_3]]
    subset_color = ["b", "r", "k"]
    ############# 改变维度 ################
    # all_out = np.reshape(all_out[:,MaxLabel], newshape=(-1, 32))
    # Y = np.reshape(test_y[:,MaxLabel], newshape=(-1, 1))
    # embedding = umap_embedding.fit_transform(X=all_out , y=Y)  # (100, 2),输入特征X，输出降维后的特征embedding
    # embedding = np.reshape(embedding, newshape=(-1, len(MaxLabel), n_components))

    embedding = umap_embedding.fit_transform(X=all_out, y=test_y.all())  # (100, 2),输入特征X，输出降维后的特征embedding
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    #embedding = np.reshape(embedding, newshape=(-1, len(MaxLabel), n_components))

    for i in range(len(MaxLabel)):
        label = MaxLabel[i]
        # plt.figure(i)
        indices = np.where(test_y[:, label] == 1.0)
        # plt.scatter(embedding[indices,i, 0], embedding[indices,i, 1], marker='.', alpha=0.5, c=test_y[:, label],label = subset_label[i])
        plt.scatter(embedding[indices, i, 0], embedding[indices, i, 1], marker='.', alpha=0.5, c=subset_color[i],label=subset_label[i])

        # plt.legend(handles=scatter.legend_elements()[0], labels=["one", "two", "three", "four", "five"])

        plt.title('UMAP Projection of Data with %s ' % functions[label])  # 多类
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
    # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], marker='.',alpha=0.5,c=Label)
    # plt.legend(*scatter.legend_elements(), loc="lower left")
    plt.show()
if __name__ == '__main__':
    main()
