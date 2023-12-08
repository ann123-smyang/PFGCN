# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # epoch,acc,loss,val_acc,val_loss
# functions='bp'
# data = pd.read_csv('res\\res_FGCN_'+functions+'.csv')
# x_axis_data1 = data["Recall"].values
# y_axis_data1 = data["Precision"].values
# data = pd.read_csv('res\\res_PGCN_'+functions+'.csv')
# x_axis_data2 = data["Recall"].values
# y_axis_data2 = data["Precision"].values
# data = pd.read_csv('res\\res_PFGCN_'+functions+'.csv')
# x_axis_data3 = data["Recall"].values
# y_axis_data3 = data["Precision"].values
#
# data = pd.read_csv('res\\res_ProtenInfer_'+functions+'.csv')
# x_axis_data4 = data["Recall"].values
# y_axis_data4 = data["Precision"].values
# data = pd.read_csv('res\\res_DeepLSTM_'+functions+'.csv')
# x_axis_data5 = data["Recall"].values
# y_axis_data5 = data["Precision"].values
# ###sep相当于上面的delimiter，是分隔符。而这个函数中也包含delimiter，它属于备用的分隔符(csv用不同的分隔符分隔数据)。
# ### header是列名，是每一列的名字，如果header=1，将会以第二行作为列名，读取第二行以下的数据。usecols同上
#
# # 画图1
# plt.figure(1)
# plt.plot(x_axis_data1, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='FAGCN')  # '
# plt.plot(x_axis_data2, y_axis_data2, 'rs--', alpha=0.3, linewidth=1, label='PAGCN')
# plt.plot(x_axis_data3, y_axis_data3, 'go--', alpha=0.3, linewidth=1, label='PFAGCN')
# plt.rcParams.update({'font.size': 19})
# plt.legend()  # 显示上面的label
#
# plt.xlabel('Recall',fontsize=19)
# plt.ylabel('Precision',fontsize=19)
#
# # plt.xlim(0.05,0.8)
# # plt.ylim(0.05,0.75)
# # plt.xlim(0.2,0.9)
# # plt.ylim(0.05,0.85)
# plt.yticks(size = 19)
# plt.xticks(size = 19)
# plt.savefig('./image/'+functions+'_PR1.eps',bbox_inches = 'tight')  # eps文件，用于LaTeX
#
# # 画图2
# plt.figure(2)
# plt.plot(x_axis_data4, y_axis_data4, 'k*--', alpha=0.5, linewidth=1, label='ProtenInfer')
# plt.plot(x_axis_data5, y_axis_data5, 'ys--', alpha=0.3, linewidth=1, label='DeepLSTM')
# plt.plot(x_axis_data3, y_axis_data3, 'go--', alpha=0.3, linewidth=1, label='PFAGCN')
# plt.rcParams.update({'font.size': 17})
# plt.legend()  # 显示上面的label
#
# plt.xlabel('Recall',fontsize=19)
# plt.ylabel('Precision',fontsize=19)
#
# # plt.xlim(0.05,0.8)
# # plt.ylim(0.05,0.8)
# # plt.xlim(0.2,0.9)
# # plt.ylim(0.05,0.85)
# plt.yticks(size = 19)
# plt.xticks(size = 19)
# plt.savefig('./image/'+functions+'_PR2.eps',bbox_inches = 'tight')  # eps文件，用于LaTeX
# # ## 设置数据标签位置及大小
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# epoch,acc,loss,val_acc,val_loss
functions='cc'
data = pd.read_csv('res\\res_PFGCN_'+functions+'_9606.csv')
x_axis_data1 = data["Recall"].values
y_axis_data1 = data["Precision"].values
data = pd.read_csv('res\\res_PFGCN_'+functions+'_10116.csv')
x_axis_data2 = data["Recall"].values
y_axis_data2 = data["Precision"].values
data = pd.read_csv('res\\res_PFGCN_'+functions+'_10090.csv')
x_axis_data3 = data["Recall"].values
y_axis_data3 = data["Precision"].values


###sep相当于上面的delimiter，是分隔符。而这个函数中也包含delimiter，它属于备用的分隔符(csv用不同的分隔符分隔数据)。
### header是列名，是每一列的名字，如果header=1，将会以第二行作为列名，读取第二行以下的数据。usecols同上

# 画图1
plt.figure(1)
plt.plot(x_axis_data1, y_axis_data1, '*--',color='firebrick', alpha=0.5, linewidth=1, label='Homo sapiens')  # '
plt.plot(x_axis_data2, y_axis_data2, 's--',color='royalblue', alpha=0.3, linewidth=1, label='Mus musculus')
plt.plot(x_axis_data3, y_axis_data3, 'o--',color='violet', alpha=0.3, linewidth=1, label='Rattus norvegicus')
plt.rcParams.update({'font.size': 19})
plt.legend()  # 显示上面的label

plt.xlabel('Recall',fontsize=19)
plt.ylabel('Precision',fontsize=19)

# plt.xlim(0.05,0.8)
# plt.ylim(0.05,0.75)
# plt.xlim(0.2,0.9)
# plt.ylim(0.05,0.85)
plt.yticks(size = 19)
plt.xticks(size = 19)
plt.savefig('./image/'+functions+'_Orgs_PR.eps',bbox_inches = 'tight')  # eps文件，用于LaTeX

plt.show()
