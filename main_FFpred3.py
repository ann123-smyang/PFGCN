#from torch_geometric.nn import GCNConv
from model.PFGCN import PFGCNgo
#from model.PFGCN import PFGCN
import time
import os
from utils_all import *
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
import numpy as np
# 导入需要的包
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
conjName=r"conj\RF.json"
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
    all_values,train_mask,val_mask,test_mask,shuffled_idx, gos, sequences,_ = load_data_mask(DATA_ROOT=args["DATA_ROOT"], FUNCTION=args["function"],
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
    X = all_values[0]
    Y = all_values[1]
    gos = np.reshape(gos,newshape=(-1,1))
    X_and_gos = np.concatenate((X, gos), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_and_gos, Y, test_size=0.1, random_state=42)
    train_gos, test_gos = X_train[:, -1], X_test[:, -1]
    X_train, X_test = X_train[:, :-1], X_test[:, :-1]
    ########### 2. 3. 训练 ###########

    # ######### 定义了新的loss函数 #########
    # #criterion = WeightedCrossEntropyLoss(args) ## loss
    # criterion = WeightedBCEWithLogitsLoss(args)  ## loss
    # ######### 更新权重 #########
    # logging.info("update the weight of loss funtion %d sec" % (time.time() - start_time))
    # class_samples = torch.tensor(np.sum(y)).to(args["device"])#.to(torch.float64)
    # criterion.update_weights(class_samples) ##更新权重

    ##### 训练 #####
    logging.info("strat training %d sec" % (time.time() - start_time))
    ################ 使用循环实现多分类器 ################
    # 一系列的SVM, 为每个功能生成一个二分类SVM
    # svm_dict = {}
    # for label in range(len(y_test[0,:])):
    #     svm = SVC(kernel='rbf', C=1.0, gamma=0.1,verbose=True)
    #     svm.fit(X_train, y_train[:, label])
    #     svm_dict[label] = svm
    #
    # # 对训练集样本进行预测
    # y_pred = []
    # for x_train in X_train:
    #     y_pred.append([svm_dict[label].predict(x_train) for label in range(len(y_test[0,:]))])
    # y_pred = torch.tensor(y_pred)

    ################ 使用循环实现多分类器 ################
    Multi_SVC = MultiOutputClassifier(SVC(kernel='rbf', C=1.0, gamma=0.1,verbose=True))
    Multi_SVC.fit(X_train, y_train)

    y_pred = Multi_SVC.predict(X_train)
    f_max, p_max, r_max, t_max, predictions_max,_,_ = compute_performance(y_pred
                                                                      , y_train
                                                                      , gos=train_gos
                                                                      , all_functions=args["functions"]
                                                                      , GO_ID=args["Go_id"], func_set=args["func_set"]
                                                                      , go=args["go"])
    roc_auc = compute_roc(y_pred, y_train, args)  # torch.Size([3141, 589]),(31530, 1) 多维数据的降维函数
    # print("the loss of testing is %f." % (test_Loss))
    print("the Fmax measure of f_max, p_max, r_max, t_max, roc_auc: \t %f %f %f %f %f." % (f_max, p_max, r_max, t_max, roc_auc))


    ############### 测试效果 ###############
    # 对训练集样本进行预测
    # y_pred = []
    # for x_test in X_test:
    #     y_pred.append([svm_dict[label].predict(x_test) for label in range(len(y_test[0, :]))])
    # y_pred = torch.tensor(y_pred)
    y_pred = Multi_SVC.predict(X_test)
    f_max, p_max, r_max, t_max, predictions_max,recall_list, predictions_list = compute_performance(y_pred
                                                                                                    , y_test
                                                                                                    , gos=test_gos
                                                                                                    , all_functions=args["functions"]
                                                                                                    , GO_ID=args["Go_id"], func_set=args["func_set"]
                                                                                                    , go=args["go"])
    roc_auc = compute_roc(y_pred, y_test,args)  # torch.Size([3141, 589]),(31530, 1) 多维数据的降维函数
    #print("the loss of testing is %f." % (test_Loss))
    print("the Fmax measure of f_max, p_max, r_max, t_max, roc_auc: \t %f %f %f %f %f." % (
    f_max, p_max, r_max, t_max, roc_auc))

    plt.plot(recall_list, predictions_list, marker='.')
    # 绘制PR曲线

    # 1. 创建文件对象
    f = open('./res/res_ffpred3_'+args["function"]+'.csv', 'w', encoding='utf-8', newline="")
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
