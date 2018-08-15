# coding=utf-8
import sys
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import jieba
import jieba.analyse
from jieba.analyse.tfidf import TFIDF


class GenerateKeyWords(object):

    def __init__(self,
                 PATH_DICT='dict.txt',
                 PATH_STOPWORDS='stopwords.txt',
                 PATH_WORD2VEC_MODEL='wiki.zh.model',
                 TOPK_TITLE=5,
                 TOPK_VOICE=10,
                 ENCODING_STOPWORDS='utf-8-sig',
                 ONLY_NV = True,
                 HMM = False):
        """
        初始化，用于加载模型和参数
        :param PATH_DICT: 用户字典的存放位置，就是整理的兴趣点
        :param PATH_STOPWORDS: 停用词的存放位置，包括之前的停用词和现在整理的停用词
        :param PATH_WORD2VEC_MODEL: 词向量模型，有四个文件
        :param TOPK_TITLE: 标题关键词的提取个数，默认5
        :param TOPK_VOICE: 声音关键词的提取个数，默认10
        :param ENCODING_STOPWORDS: 停用词的编码，默认无标签的utf-8
        :param ONLY_NV: 是否进行词性筛选，默认是，保留名词和动词时的效果较好，
        ’x'为未知词性，用户自定义添加的词典里面有些词是未知词性，需要保留
        :param HMM: 分词时是否使用hmm算法，不使用hmm时效果好，hmm在结巴中主要用于新词发现
        """
        print('load')
        self._path_dict = PATH_DICT
        self._path_stopwords = PATH_STOPWORDS
        self._path_words2vec = PATH_WORD2VEC_MODEL

        jieba.load_userdict(self._path_dict)
        with open(self._path_stopwords, 'r', encoding=ENCODING_STOPWORDS) as sw:
            self._stop_words = [line.strip() for line in sw.readlines()]
        self._idf, _ = TFIDF().idf_loader.get_idf()
        self._model = Word2Vec.load(self._path_words2vec)
        self._topk_title = TOPK_TITLE
        self._topk_voice = TOPK_VOICE
        self._only_nv = ONLY_NV
        self._hmm = HMM
        self.predicted_poi = []
        print('model load success!')

    def __simi(self,word1, word2):
        """
        词语相似度计算
        :param word1:
        :param word2:
        :return: 相似度 float64
        """
        try:
            value = 0.0
            if word1 in self._model and word2 in self._model:
                value = self._model.similarity(word1, word2)
            return value
        except:
            return 0.0

    def __comput_tfidf(self,sentence, topk=10):
        """
        提取句子中的关键词，算法tf-idf
        :param sentence: 分词后的句子，字符串类型，词语之间逗号分隔
        :param number: 需要提取的关键词的个数
        :return: list 排序后的关键词
        """
        words = sentence.split(',')
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0.0) + 1.0

        totle = sum(freq.values())
        dict_weights = {}
        for w in freq.keys():
            weight = self._idf.get(w, 0.) * freq[w] / totle
            dict_weights[w] = weight

        if len(dict_weights) < 1:
            return []

        sorted_pair = sorted(dict_weights.items(), key=lambda x: x[1], reverse=True)
        new_words, _ = zip(*sorted_pair)

        topk = min(len(new_words), topk)
        return new_words[:topk]

    def __jieba_cut(self,sentence,n_v=True, HMM=True):
        """
        jieba分词,去停用词
        :param sentence:句子
        :param n_v: 是否词性筛选，仅保留名词动词
        :param HMM:
        :return: string e.g.  'ac,adsa,qwe’
        """
        if n_v:
            voice_tok = jieba.posseg.cut(sentence,HMM)
            tok_list = []
            for w in voice_tok:
                if w not in self._stop_words:  # 去停用词
                    if w.flag[0] == 'n' or w.flag[0] == 'v' or w.flag[0] == 'x':  # 仅保留动词和名词，部分自定义的词词性为'x'
                        tok_list.append(w.word)
            return ','.join(tok_list)
        else:
            sentence_tok = jieba.cut(sentence,HMM)
            tok_list = []
            for w in sentence_tok:
                if w not in self._stop_words:  # 去停用词
                    tok_list.append(w.word)
            return ','.join(tok_list)

    def __list_clean(self, sentence_list):
        """
        去重，同时去掉词长度为1的词,保留原始顺序
        :param sentence_list: [...]
        :return: [...]
        """
        words_dict = {}
        iterator = 0
        for w in sentence_list:
            w = w.strip()
            if w not in words_dict and len(w) > 1:
                words_dict[w] = iterator
                iterator += 1

        sorted_w = sorted(words_dict.items(), key=lambda x: x[1], reverse=False)
        words, _ = list(zip(*sorted_w))
        return words

    def generate(self, title, voice):
        """
        根据title和voice两个字符串生成最终的关键词
        :param title: title文本，字符串类型
        :param voice: 声音文本，字符串类型
        :return: 无返回，将最终的结果保存到类的属性里面，那么关键字的属性为：self.predicted_poi
        """
        # 分词
        title_cut = self.__jieba_cut(title, self._only_nv, self._hmm)
        voice_cut = self.__jieba_cut(voice, self._only_nv, self._hmm)
        # 提取关键词
        important_title_words = self.__comput_tfidf(title_cut, self._topk_title)
        important_voice_words = self.__comput_tfidf(voice_cut, self._topk_voice)
        # 如果title 和 voice 的关键词为空，异常处理
        if len(important_title_words) < 1:
            self.predicted_poi = important_voice_words
        elif len(important_voice_words) < 1:
            self.predicted_poi = important_title_words
        elif len(important_title_words) < 1 and len(important_voice_words) < 1:
            self.predicted_poi = []
        else:
            # 正常状态
            simi_dict = {}
            # 计算相似度
            for w1 in important_title_words:
                for w2 in important_voice_words:
                    simi_value = self.__simi(w1, w2)
                    simi_dict[(w1, w2)] = simi_value

            # 相似度排序
            sorted_pairs = sorted(simi_dict.items(), key=lambda x: x[1], reverse=True)
            big_pairs, _ = zip(*sorted_pairs)

            # 去重得到最终结果
            important_words = []
            for tuple_p in big_pairs:
                important_words.append(tuple_p[0])
                important_words.append(tuple_p[1])
            self.predicted_poi = self.__list_clean(important_words)


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print('参数错误')
        sys.exit(1)
    title = sys.argv[1]
    voice = sys.argv[2]

    poi = GenerateKeyWords()
    poi.generate(title, voice)
    print(poi.predicted_poi)

