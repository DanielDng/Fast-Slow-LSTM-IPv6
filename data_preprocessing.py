import sys
sys.path.append("/path/to/sklearn")
from sklearn import preprocessing
import numpy as np
import csv

# add at 2018-1-12
# all the value of kdd99

def countingFunction(type_into, name):
    s_c = 1
    f_c = 1
    p_c = 1
    protocol_type = ["ICMPv6","IPv6","TCP","UDP"]
    # example TCP->1
    # example UDP->2
    # example ICMP->3
    Source = ["fe80::494:b5a9:b979:6592","fe80::4dc8:22c0:eb62:e540","fe80::cd98:c5e0:3f6c:90c0",
    "fe80::3680:b3ff:fe42:d357","ff02::1:ff42:d357","ff02::1:ff6c:90c0"]
    # example aol->1
    # example Z39_50->70
    Destination = ["fe80::494:b5a9:b979:6592","fe80::4dc8:22c0:eb62:e540","fe80::cd98:c5e0:3f6c:90c0",
    "fe80::3680:b3ff:fe42:d357","ff02::1:ff42:d357","ff02::1:ff6c:90c0"]
    # example OTH->1
    # example SH->11

    if type_into == 2:
        for name_s in Source:
            if name == name_s:
                return s_c
            else:
                s_c += 1

    elif type_into == 3:
        for name_f in Destination:
            if name == name_f:
                return f_c
            else:
                f_c += 1

    elif type_into == 4:
        for name_p in protocol_type:
            if name_p == name:
                return p_c
            else:
                p_c += 1


def replace_kdd(into_string):
    cc_num = 1
    want_num = 0
    replace_list = []
	# add at 2018-1-12
	# PLEASE keep and do NOT change the 99999 here
	# you can edit the 2, 3, 4 value or more
	# this is bug but I don't want to fix it :/
    want_replace = [2, 3, 4, 99999]

    for i in into_string:
        if cc_num == want_replace[want_num]:
            get_the_value = countingFunction(cc_num, i)
            #print(i, get_the_value)
            replace_list.append(str(get_the_value))
            cc_num += 1
            want_num += 1
        else:
            replace_list.append(i)
            cc_num += 1
    #rep = ",".join(replace_list)
    #return rep
    return replace_list


def load_kdd():
    # load the kdd from file(add at 2018/1/12)
	#You_file_path = raw_input(
	#	"Please enter you kdd file path and name here: \n")
	#Name_you_want_to_save = raw_input(
	#    "Please enter the file name you want to save: \n")
    You_file_path = "data/ipv6_test_orial.csv"
    train_filename='data/ipv6_train.csv'
    Name_you_want_to_save = "data/ipv6_test_orial_o.csv"
    try:
        kdd_read = open(You_file_path, "r")
    except IOError as e:
        print('Can not open the ipv6 file')
        print(e)
        sys.exit(1)

    para_num = 5
    
    kdd_readlines = kdd_read.readlines()
    test_line_num = len(kdd_readlines)
    test_orial = np.zeros((test_line_num, para_num+1))
    j = 0
    for k in kdd_readlines:
        k_split = k.split(",")
        k_over = replace_kdd(k_split)
        print(k_over)
        test_orial[j,:] = k_over
        j = j + 1
    kdd_read.close()

    fr1 = open(train_filename)
    print ('load training data from %s',train_filename)
    lines1 = fr1.readlines()
    line_nums1 = len(lines1)
    traindata = np.zeros((line_nums1, para_num+1))
    for i in range(line_nums1):
        line = lines1[i].strip()
        traindata[i, :] = line.split(',')
    fr1.close()
    _traindata= traindata[ :, 0:5]

    min_max_scaler = preprocessing.MinMaxScaler()
    train_minmax = min_max_scaler.fit_transform(_traindata)
    dataset_p_minmax = min_max_scaler.transform(test_orial[:, 0:5])

    train_scaled = preprocessing.scale(train_minmax)
    scaler = preprocessing.StandardScaler().fit(train_minmax)
    test_scaled = scaler.transform(dataset_p_minmax)

    i = 0
    with open(You_file_path) as csvfile:
        rows = csv.reader(csvfile)
        with open(Name_you_want_to_save,'w', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                row.append(test_scaled[i,0])
                row.append(test_scaled[i,1])
                row.append(test_scaled[i,2])
                row.append(test_scaled[i,3])
                writer.writerow(row)
                i += 1

def main():
    # main function here(add at 2018/1/12)
    load_kdd()


if __name__ == "__main__":
    main()