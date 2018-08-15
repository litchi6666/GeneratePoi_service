import math
import pandas as pd
import numpy as np


def abc(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return (precision * recall) / (precision + recall)


def eval_pridict(y_input,y_predict):
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    right_label_num = 0  # 总命中标签数量
    length = len(y_input)
    for i in range(length):
        input = y_input[i]
        predict = y_predict[i]
        if input is np.nan or predict is np.nan:
            continue


        sample_num += len(predict)
        all_marked_label_num += len(input)

        for w in predict.split(','):
            if w in input.split(','):
                right_label_num += 1

    presicion = float(right_label_num)/float(sample_num)
    recall = float(right_label_num)/float(all_marked_label_num)

    score = (presicion * recall) / (presicion + recall)

    return presicion, recall, score


def eval_pridict_2(y_input,y_predict):

    def mean(list):
        sum = 0.0
        for i in list:
            sum += i
        return sum/len(list)

    length = len(y_input)

    p_list,r_list,f_list =[],[],[]

    for i in range(length):
        input = y_input[i]
        predict = y_predict[i]
        if input is np.nan or predict is np.nan:
            continue
        right_label_num = 0  # 总命中标签数量


        sample_num = len(predict)
        all_marked_label_num = len(input)

        for w in predict.split(','):
            if w in input.split(','):
                right_label_num += 1

        presicion = float(right_label_num)/float(sample_num)
        recall = float(right_label_num)/float(all_marked_label_num)

        if right_label_num == 0:
            score = 0
        else:
            score = (presicion * recall) / (presicion + recall)
        p_list.append(presicion)
        r_list.append(recall)
        f_list.append(score)

    p = mean(p_list)
    r = mean(r_list)
    f = mean(f_list)

    return p, r, f


def process(path):
    ff = pd.read_csv(path, encoding='utf-8-sig')
    y_input = ff['poi']
    y_predict = ff['predict']

    presicion, recall,score = eval_pridict(y_input, y_predict)
    print(presicion, recall, score)

    p,r,f = eval_pridict_2(y_input, y_predict)
    print(p,r,f)


process('naive_poi_old.csv')
print('*'*20)
process('naive_poi_nv.csv')
print('*'*20)
process('naive_poi_n.csv')
print('*'*20)
process('naive_poi_nohmm_nv.csv')
print('*'*20)
process('naive_poi_nohmm_n.csv')
print('*'*20)
process('naive_poi_stop_nv.csv')
print('*'*20)
process('naive_poi_stop_nv__new.csv')