#from torch_geometric.nn import GCNConv
from model.DLSTM import DeepLSTM
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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
conjName=r"conj\DLSTM.json"
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
    all_values, train_mask, val_mask, test_mask, shuffled_idx, gos, sequences, _, max_sequences = load_sequences( DATA_ROOT=args["DATA_ROOT"],
                                                                                                   FUNCTION=args["function"],
                                                                                                   ORG=args["org"], MAXLEN=args["MAXLEN"])
    #data_gos = all_values['gos'].values
    logging.info("Data loaded in %d sec" % (time.time() - start_time))

    ########################## 加载数据为batch ########################
    x = all_values[0]
    y = all_values[1]
    ### add
    args["functions"] = args["functions"][:y.shape[1]]
    args["nb_classes"] = y.shape[1]
    args["MAXLEN"] = max_sequences
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

    ########### 2. 3. 训练 ###########
    model = DeepLSTM(args).to(args["device"])
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
    for epoch in range(args["epochs"]):
        i = 0
        train_Loss = 0
        #Ap = 0
        all_out = torch.empty((0, args["nb_classes"])).to(args["device"])
        for batch,the_train_mask,idx in zip(dataloader, train_mask_loader,idex_loader):
            batch_x,batch_y = batch[0][:,:-1*args["nb_classes"]] ,batch[0][:,-1*args["nb_classes"]:]
            model.train()

            optimizer.zero_grad()
            out = model(batch_x)#torch.Size([64, 589])
            out = out[the_train_mask[0], :]
            batch_y = batch_y[the_train_mask[0],:]
            loss = criterion(out, batch_y)  ##新loss函数
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
            torch.save(model.state_dict(), f"experiments\\DeepLSTM\\best_model_{best_epoch}.pth")
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
    print("the best epoch is",best_epoch)
    best_model = DeepLSTM(args)
    best_model.load_state_dict(torch.load(f"experiments\\DeepLSTM\\best_model_{best_epoch}.pth"))

    ########################## 在测试集上进行预测 ##########################
    i = 0
    test_Loss = 0
    #Ap = 0
    best_model = best_model.to(args["device"])
    all_out = torch.empty((0,args["nb_classes"])).to(args["device"])
    test_y = torch.empty((0,args["nb_classes"])).to(args["device"])#torch.Size([1, 589])
    test_gos_all = gos[test_mask.detach().cpu().numpy()]
    best_model.eval()
    for batch, the_test_mask , idx in zip(dataloader, test_mask_loader,idex_loader):
        batch_x, batch_y = batch[0][:, :-1 * args["nb_classes"]], batch[0][:, -1 * args["nb_classes"]:]
        if batch_x.shape[0]!= args["batch_size"]:
            continue
        out = best_model(batch_x)  # torch.Size([64, 589])
        out = out[the_test_mask[0], :]
        batch_y = batch_y[the_test_mask[0],:]
        loss = criterion(out, batch_y)  ##新loss函数
        #loss = F.cross_entropy(out, batch_y.long())
        #loss = F.nll_loss(out, batch_y.long())
        test_gos = gos[idx[0].cpu()][test_mask[0].cpu()]

        if batch_y.shape[0]==0:
            continue
        # ap = accuracy_score(y_true=batch_y.detach().cpu().numpy(),
        #                     y_pred=np.reshape(np.argmax(out.detach().cpu().numpy(), axis=1),
        #                     newshape=(-1, 1)))
        test_Loss = test_Loss + loss
        #Ap = Ap + ap
        all_out = torch.concat((all_out, out), dim=0)
        test_y = torch.concat((test_y,batch_y), dim=0)
        i = i + 1

    f_max, p_max, r_max, t_max, predictions_max,recall_list,predictions_list = compute_performance(all_out.detach().cpu().numpy()
                                                                      , test_y.detach().cpu().numpy()
                                                                      , gos=test_gos_all
                                                                      , all_functions=args["functions"]
                                                                      , GO_ID=args["Go_id"], func_set=args["func_set"]
                                                                      , go=args["go"])
    roc_auc = compute_roc(all_out.detach().cpu().numpy(), test_y.detach().cpu().numpy(), args)  # torch.Size([3141, 589]),(31530, 1) 多维数据的降维函数
    print("the loss of testing is %f." % (test_Loss))
    print("the Fmax measure of f_max, p_max, r_max, t_max, roc_auc: \t %f %f %f %f %f." % (f_max, p_max, r_max, t_max,roc_auc))


    plt.plot(recall_list,predictions_list, marker='.')
    # 绘制PR曲线

    # 1. 创建文件对象
    f = open('res/res_DeepLSTM_'+args["function"]+'.csv', 'w', encoding='utf-8', newline="")
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    csv_writer.writerow(["Recall", "Precision"])

    # 4. 写入csv文件内容

    for i in range(len(predictions_list)):
        csv_writer.writerow([recall_list[i], predictions_list[i]])
    # 5. 关闭文件
    f.close()

    ############ 画图  ############
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
if __name__ == '__main__':
    main()




