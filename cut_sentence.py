"""
实现句子的分词
注意点：
1. 实现单个字分词

2. 实现按照词语分词
2.1 加载词典

3. 使用停用词
"""

import string
import jieba
import jieba.posseg as psg
# from utils import stopwords
import logging

#关闭jieba日志
jieba.setLogLevel(logging.INFO)

#加载词典
# jieba.load_userdict("./corpus/keywords.txt")

stopwords = []
continue_words = string.ascii_lowercase

def _cut_sentence_by_word(sentence):
    """
    按照单个字进行分词,eg:python 可 以 做 人 工 智 能 么 ？ jave
    :param sentence:str
    :return: [str,str,str]
    """
    temp = ""
    result = []
    for word in sentence:
        if word in continue_words:
            temp += word
        else:
            if len(temp)>0:
                result.append(temp)
                temp = ""
            result.append(word)

    if len(temp)>0:
        result.append(temp)
    return result


def _cut_sentence(sentence,use_stopwords,use_seg):
    """
    按照词语进行分词
    :param sentence:str
    :return: 【str,str,str】
    """
    if not use_seg:
        result = jieba.lcut(sentence)
    else:
        result = [(i.word,i.flag) for i in psg.cut(sentence)]
    if use_stopwords:
        if not use_seg:
            result = [i for i in result if i not in stopwords]
        else:
            result = [i for i in result if i[0] not in stopwords]
    return result

def cut(sentence,by_word=False,use_stopwords=False,use_seg=False):
    """
    封装上述的方法
    :param sentence:str
    :param by_word: bool，是否按照单个字分词
    :param use_stopwords: 是否使用停用词
    :param use_seg: 是否返回词性
    :return: [(str,seg),str]
    """
    sentence = sentence.lower()
    if by_word:
        return _cut_sentence_by_word(sentence)
    else:
        return _cut_sentence(sentence,use_stopwords,use_seg)
