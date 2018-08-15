import pandas as pd
import jieba
import jieba.posseg

to_cut_file = 'D:\\data\\voice\\video_voice_clean.csv'
to_cut = pd.read_csv(to_cut_file,encoding='utf-8-sig',header=0)

poi_file = 'D:\\data\\videos\\videos.csv'
poi = pd.read_csv(poi_file, encoding='utf-8-sig', header=0)

voice_cut = 'D:\\data\\voice\\voice_cut_n___.csv'


def stop_words():
    '''获取并且返回停用词表'''
    stopwords = []
    with open('d:/data/stopwords.txt', 'r', encoding='utf-8-sig') as ff:
        lines = ff.readlines()
        for l in lines:
            stopwords.append(l.strip())
    print(len(stopwords))
    return stopwords


def list_clean(sentence_list):
    words_dict = {}
    iterator = 0
    for w in sentence_list:
        w = w.strip()
        if w not in words_dict:
            words_dict[w] = iterator
            iterator += 1
    sorted_w = sorted(words_dict.items(), key=lambda x: x[1], reverse=False)
    words, _ = list(zip(*sorted_w))
    words_dict.clear()
    return words


def jieba_cut(sentence,n_v=True):
    if n_v:
        voice_tok = jieba.posseg.cut(sentence)
        tok_list = []
        for w in voice_tok:
            if w.flag[0] == 'n' or w.flag[0] == 'x':  # 仅保留动词和名词   or w.flag[0] == 'v'
                tok_list.append(w.word)
        return ','.join(tok_list)
    else:
        return ','.join(jieba.cut(sentence))


stopwords = stop_words()

video_ids = poi['video_id'].tolist()


id_list,title_list,poi_list,voice_cut_list = [],[],[],[]

jieba.load_userdict('d:/data/dict.txt')

for i in range(len(to_cut['id'])):
    try:
        id = to_cut['id'][i]
        voice = to_cut['voice'][i]

        index = video_ids.index(id)
        title = poi['title'][index]
        poi_ = list_clean(poi['poi'][index].replace('"','').split(','))

        voice_tok = voice #jieba_cut(voice, True)

        id_list.append(id)
        title_list.append(title)
        poi_list.append(','.join(poi_))
        voice_cut_list.append(voice_tok)

        if i%100 == 0:
            print(i)
    except:
        print("==============",i)
save = pd.DataFrame({'id':id_list,'title':title_list,'poi':poi_list,'voice':voice_cut_list},columns=['id','title',
                                                                                                     'poi','voice'])
save.to_csv(voice_cut,',',encoding='utf-8')

