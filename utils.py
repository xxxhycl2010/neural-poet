# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/29
# Brief: 

import collections


def get_logger(name=None):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('app.log')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = get_logger('neural-poet')


def get_train_data(paths):
    """
    Load train data set
    :param path:
    :return: list
    """
    train_data = []
    logger.info("[reader] load train data set from %s" % ";".join(paths))
    for train_path in paths:
        train_data.extend(load_train_dict(train_path))
    return train_data


def load_train_dict(dic_path):
    """
    Load train dict file
    :param dic_path:
    :return:
    """
    poetrys = []
    with open(dic_path, "r", encoding='utf-8', ) as f:
        for line in f:
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poetrys.append(content)
            except Exception as e:
                logger.error("file read error, %s %s" % (dic_path, e))

    # 按诗的字数排序
    poetrys = sorted(poetrys, key=lambda line: len(line))
    logger.info('%s: poetry size: %d' % (dic_path, len(poetrys)))
    return poetrys


def word_count(data):
    # 统计每个字出现次数
    all_words = []
    for poetry in data:
        all_words += [word for word in poetry]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    return words
