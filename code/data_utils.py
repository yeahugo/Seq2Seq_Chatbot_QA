#coding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
__author__ = u'qhduan@memect.co'
import os
import sys
import json
import math
import shutil
try:
    import pickle
except ImportError:
    import six.moves.cPickle as pickle
import sqlite3
from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm


def with_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, p)


DICTIONARY_PATH = u'db/dictionary.json'
EOS = u'<eos>'
UNK = u'<unk>'
PAD = u'<pad>'
GO = u'<go>'
buckets = [(5, 15), (10, 20), (15, 25), (20, 30)]


def time(s):
    ret = u''
    if (s >= (60 * 60)):
        h = math.floor((s / (60 * 60)))
        ret += u'{}h'.format(h)
        s -= ((h * 60) * 60)
    if (s >= 60):
        m = math.floor((s / 60))
        ret += u'{}m'.format(m)
        s -= (m * 60)
    if (s >= 1):
        s = math.floor(s)
        ret += u'{}s'.format(s)
    return ret


def load_dictionary():
    with open(with_path(DICTIONARY_PATH), u'r') as fp:
        dictionary = ([EOS, UNK, PAD, GO] + json.load(fp))
        index_word = OrderedDict()
        word_index = OrderedDict()
        for (index, word) in enumerate(dictionary):
            index_word[index] = word
            word_index[word] = index
        dim = len(dictionary)
    return (dim, dictionary, index_word, word_index)


u"\ndef save_model(sess, name='model.ckpt'):\n    import tensorflow as tf\n    if not os.path.exists('model'):\n        os.makedirs('model')\n    saver = tf.train.Saver()\n    saver.save(sess, with_path('model/' + name))\n\ndef load_model(sess, name='model.ckpt'):\n    import tensorflow as tf\n    saver = tf.train.Saver()\n    saver.restore(sess, with_path('model/' + name))\n"
(dim, dictionary, index_word, word_index) = load_dictionary()
print(u'dim: ', dim)
EOS_ID = word_index[EOS]
UNK_ID = word_index[UNK]
PAD_ID = word_index[PAD]
GO_ID = word_index[GO]


class BucketData(object):

    def __init__(self, buckets_dir, encoder_size, decoder_size):
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.name = (u'bucket_%d_%d.db' % (encoder_size, decoder_size))
        self.path = os.path.join(buckets_dir, self.name)
        self.conn = sqlite3.connect(self.path)
        self.cur = self.conn.cursor()
        sql = u'SELECT MAX(ROWID) FROM conversation;'
        self.size = self.cur.execute(sql).fetchall()[0][0]

    def all_answers(self, ask):
        u'找出所有数据库中符合ask的answer\n        '
        sql = u"\n        SELECT answer FROM conversation\n        WHERE ask = '{}';\n        ".format(
            ask.replace(u"'", u"''"))
        ret = []
        for s in self.cur.execute(sql):
            ret.append(s[0])
        return list(set(ret))

    def random(self):
        while True:
            rowid = np.random.randint(1, (self.size + 1))
            sql = u'\n            SELECT ask, answer FROM conversation\n            WHERE ROWID = {};\n            '.format(
                rowid)
            ret = self.cur.execute(sql).fetchall()
            if (len(ret) == 1):
                (ask, answer) = ret[0]
                if ((ask is not None) and (answer is not None)):
                    return (ask, answer)


def read_bucket_dbs(buckets_dir):
    ret = []
    for (encoder_size, decoder_size) in buckets:
        bucket_data = BucketData(buckets_dir, encoder_size, decoder_size)
        ret.append(bucket_data)
    return ret


def sentence_indice(sentence):
    ret = []
    for word in sentence:
        if (word in word_index):
            ret.append(word_index[word])
        else:
            ret.append(word_index[UNK])
    return ret


def indice_sentence(indice):
    ret = []
    for index in indice:
        word = index_word[index]
        if (word == EOS):
            break
        if ((word != UNK) and (word != GO) and (word != PAD)):
            ret.append(word)
    return u''.join(ret)


def vector_sentence(vector):
    return indice_sentence(vector.argmax(axis=1))


def generate_bucket_dbs(input_dir, output_dir, buckets, tolerate_unk=1):
    pool = {

    }
    word_count = Counter()

    def _get_conn(key):
        if (key not in pool):
            if (not os.path.exists(output_dir)):
                os.makedirs(output_dir)
            name = (u'bucket_%d_%d.db' % key)
            path = os.path.join(output_dir, name)
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute(
                u'CREATE TABLE IF NOT EXISTS conversation (ask text, answer text);')
            conn.commit()
            pool[key] = (conn, cur)
        return pool[key]
    all_inserted = {

    }
    for (encoder_size, decoder_size) in buckets:
        key = (encoder_size, decoder_size)
        all_inserted[key] = 0
    db_paths = []
    for (dirpath, _, filenames) in os.walk(input_dir):
        for filename in (x for x in sorted(filenames) if x.endswith(u'.db')):
            db_path = os.path.join(dirpath, filename)
            db_paths.append(db_path)
    for db_path in db_paths:
        print(u'读取数据库: {}'.format(db_path))
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        def is_valid(s):
            unk = 0
            for w in s:
                if (w not in word_index):
                    unk += 1
                    if (unk > tolerate_unk):
                        return False
            return True
        total = c.execute(
            u'SELECT MAX(ROWID) FROM conversation;').fetchall()[0][0]
        ret = c.execute(u'SELECT ask, answer FROM conversation;')
        wait_insert = []

        def _insert(wait_insert):
            if (len(wait_insert) > 0):
                for (encoder_size, decoder_size, ask, answer) in wait_insert:
                    key = (encoder_size, decoder_size)
                    (conn, cur) = _get_conn(key)
                    cur.execute(u"\n                    INSERT INTO conversation (ask, answer) VALUES ('{}', '{}');\n                    ".format(
                        ask.replace(u"'", u"''"), answer.replace(u"'", u"''")))
                    all_inserted[key] += 1
                for (conn, _) in pool.values():
                    conn.commit()
                wait_insert = []
            return wait_insert
        for (ask, answer) in tqdm(ret, total=total):
            if (is_valid(ask) and is_valid(answer)):
                for i in range(len(buckets)):
                    (encoder_size, decoder_size) = buckets[i]
                    if ((len(ask) <= encoder_size) and (len(answer) < decoder_size)):
                        word_count.update(list(ask))
                        word_count.update(list(answer))
                        wait_insert.append(
                            (encoder_size, decoder_size, ask, answer))
                        if (len(wait_insert) > 10000000):
                            wait_insert = _insert(wait_insert)
                        break
    word_count_arr = [(k, v) for (k, v) in word_count.items()]
    word_count_arr = sorted(word_count_arr, key=(lambda x: x[1]), reverse=True)
    wait_insert = _insert(wait_insert)
    return (all_inserted, word_count_arr)


if (__name__ == u'__main__'):
    print(u'generate bucket dbs')
    db_path = u''
    if ((len(sys.argv) >= 2) and os.path.exists(sys.argv[1])):
        db_path = sys.argv[1]
        if (not os.path.isdir(db_path)):
            print(u'invalid db source path, not dir')
            exit(1)
    elif os.path.exists(u'./db'):
        db_path = u'./db'
    else:
        print(u'invalid db source path')
        exit(1)
    target_path = u'./bucket_dbs'
    if (not os.path.exists(target_path)):
        os.makedirs(target_path)
    elif (os.path.exists(target_path) and (not os.path.isdir(target_path))):
        print(u'invalid target path, exists but not dir')
        exit(1)
    elif (os.path.exists(target_path) and os.path.isdir(target_path)):
        shutil.rmtree(target_path)
        os.makedirs(target_path)
    (all_inserted, word_count_arr) = generate_bucket_dbs(
        db_path, target_path, buckets, 1)
    for (key, inserted_count) in all_inserted.items():
        print(key)
        print(inserted_count)
    print(u'done')
