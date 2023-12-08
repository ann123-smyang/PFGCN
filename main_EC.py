from Bio import ExPASy
from Bio import SwissProt
from urllib.error import URLError
import ssl
import urllib.request
# 禁用requests出现的取消ssl验证的警告，直接引用如下
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import requests
import time
from main_PFGCN import read_config
import pandas as pd
from utils_all import get_Uniprot_ID
import re
ssl._create_default_https_context = ssl._create_unverified_context

def get_ec_numbers(protein_id):
    """从Uniprot id获取蛋白质的EC号
    Args:
        protein_id (str): Uniprot的蛋白质accession number
    Returns:
        set: 蛋白质的EC号集合
    """
    try:
        #http://www.uniprot.org/uniprot/XXX.txt
        handle = ExPASy.get_sprot_raw(protein_id)
        record = SwissProt.read(handle)

    except URLError as e:
        print(f'网络错误:{e}, 未能获取{protein_id}的信息')
        return set()

    ############## 遍历得到 ##############
    print([x for x in record.cross_references])
    ecs = set([x[2] for x in record.cross_references if x[0] == 'EC'])
    ######################### 使用正则表达式匹配EC编号 #########################
    #pattern = r'EC; EC (\d+(?:\.\d+){3})'
    # pattern = r'EC (\d+(?:\.\d+){3})'
    # #ECO: 0000255
    # #pattern = r'ECO: (\d+(?:\.\d+){3})'
    # pattern = r'GO:\d+'#GO:0033644
    #
    # text = record
    # # 查找所有匹配,返回匹配对象列表 ,
    # matches = re.findall(pattern, str(text))
    # print(text)
    # # 从匹配中提取并打印EC号码
    # ec_numbers = [m.group(1) for m in matches]
    # print(ec_numbers)
    return ecs

if __name__ == '__main__':
    protein_id = 'Q6GZX3'#Uniprot ID,Q6GZX3,Q197F8
    ecs = get_ec_numbers(protein_id)
    print(f'{protein_id}的EC号为:{ecs}')
    args = read_config()
    func_df = pd.read_pickle(args["DATA_ROOT"] + args["function"] + '.pkl')
    functions = func_df['functions'].values
    nb_classes = len(functions)
    args["nb_classes"] = nb_classes
    start_time = time.time()
    print("function:", args["function"])
    protein_ids = get_Uniprot_ID(DATA_ROOT=args["DATA_ROOT"],
                                 FUNCTION=args["function"],
                                 ORG=args["org"], MAXLEN=args["MAXLEN"])
    # 忽略证书验证
    # ssl_context = ssl.create_default_context()
    # ssl._create_default_https_context = ssl._create_unverified_context()
    # ssl_context.check_hostname = False
    # ssl_context.verify_mode = ssl.CERT_NONE
    # requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    for protein_id in protein_ids:
        ecs = get_ec_numbers(protein_id)
        print(f'{protein_id}的EC号为:{ecs}')
