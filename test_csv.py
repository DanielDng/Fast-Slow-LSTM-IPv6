import csv
import pandas as pd

with open('/home/dl/lx/Fast-Slow-LSTM-master_ui/data/ipv6_log.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column1 = [row[1]for row in reader]
    print(column1)


# # 下面是按照列属性读取的
# d = pd.read_csv('D:\Data\新建文件夹\list3.2.csv', usecols=['case', 'roi', 'eq. diam.','x loc.','y loc.','slice no.'])
# print(d)

# d = pd.read_csv('D:\Data\新建文件夹\list3.2.csv', usecols=['case', 'roi', 'eq. diam.','x loc.','y loc.','slice no.'],
#                 nrows=10)