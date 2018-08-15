import pandas as pd
import numpy as np


def load_words():
    path = 'file/a.csv'
    ff = pd.read_csv(path,encoding='utf-8-sig')
    dict_words = {}
    interator = 0
    for i in range(len(ff['标签ID'])):
        poi = ff['兴趣点名称'][i]
        poi_type_1 = ff['兴趣点类型1'][i]
        poi_type_2 = ff['兴趣点类型1'][i]

        if poi_type_1 not in dict_words:
            dict_words[poi_type_1] = interator
            interator += 1

        if poi_type_2 not in dict_words:
            dict_words[poi_type_2] = interator
            interator += 1

        if poi not in dict_words:
            dict_words[poi] = interator
            interator += 1

    pairs = sorted(dict_words.items(), key=lambda x: x[1])
    words,_ = zip(*pairs)

    return words


def save_words():
    words = load_words()
    with open('d:/data/dict.txt','w',encoding='utf-8') as ff:
        try:
            for w in words:
                ff.write(w+' 5 n')
                print(w)
                ff.write('\n')
        except:
            pass
    print('done!')


if __name__ == '__main__':
    save_words()