from Bio import ExPASy
from Bio import SwissProt
from urllib.error import URLError
from utils_all import get_Uniprot_ID
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
Data_path = "D:\机器学习与深度学习\论文收集\数据集\Swiss- Prot数据集-蛋白质功能预测\\uniprot_sprot.tab"
class TabFile:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def read(self):
        with open(self.filename, self.mode) as f:
            data = []
            for line in f:
                row = line.strip().split('\t')
                data.append(row)
        return data

    def write(self, data):
        with open(self.filename, self.mode) as f:
            for row in data:
                line = '\t'.join([str(x) for x in row])
                f.write(line + '\n')

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
    tf = TabFile(Data_path, 'r')
    data = tf.read()
    # sequeece = data[:,0]
    # sequeece = data[:,1]
    print(type(data))
    print(len(data))
    Sequeeze = []
    Uniprot_ID = []

    for i in range(len(data)):
        uniprot_ID = data[i][0][3:9]
        sequeeze = data[i][1]
        Ec = get_ec_numbers(uniprot_ID)
        print(Ec)