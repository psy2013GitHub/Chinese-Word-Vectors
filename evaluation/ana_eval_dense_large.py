# We reuse a fraction of code in http://bitbucket.org/omerlevy/hyperwords.
# Using the numpy and similarity matrix largely speed up the evaluation process,
# compared with evaluation scripts in word2vec and GloVe

import gc
import time
import numpy as np
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm


DEFAULT_SEP = '\t'

from functools import wraps
def print_func(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        print('{} ...'.format(fun.__name__))
        return fun(*args, **kwargs)
    return wrapper

def time_cost(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = fun(*args, **kwargs)
        print('cost time: {} {}'.format(fun.__name__, time.time() - start))
        return ret
    return wrapper

@time_cost
@print_func
def build_vocab(path, iw=None, header=True, dim=None, sep=DEFAULT_SEP, build_all_vocab=False):
    '''
        在已有词表iw基础上继续建立词典
    :param path:
    :return:
    '''
    lines_num = 0
    new_iw = [] if iw is None else iw.copy()
    iw_seen_set = set()
    wi_tmp = {w:i for i,w in enumerate(new_iw)}
    new_wi = wi_tmp.copy()
    analogy_matrix = None
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in tqdm(f):
            if first_line:
                if header:
                    _dim = int(line.rstrip().split()[1])
                    if dim:
                        assert dim == _dim, '{} != {}'.format(_dim, dim)
                    analogy_matrix = np.zeros([len(iw), dim], dtype=np.float32)
                    first_line = False
                    continue

            line = line.rstrip()
            if not line:
                continue

            tokens = line.rsplit(sep, dim)

            if first_line:
                analogy_matrix = np.zeros([len(iw), dim])

            w = tokens[0]
            if not w in new_wi and build_all_vocab:
                new_iw.append(w)
                new_wi[w] = len(new_iw) - 1

            if iw and w in wi_tmp:
                iw_seen_set.add(w)
                analogy_matrix[wi_tmp[w]] = np.asarray([float(x) for x in tokens[1:]])

            lines_num += 1
            first_line = False if first_line else False

            if lines_num % 100000 == 0:
                gc.collect()

    unseen_words = set(iw).difference(iw_seen_set) if iw else set()

    return new_iw, new_wi, dim, analogy_matrix, unseen_words

@time_cost
@print_func
def read_vectors(path, chunk_size=None, dim=None, header=True, sep=DEFAULT_SEP, error='warn'):
    lines_num = 0
    vectors, words = [], []
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                if header:
                    _dim = int(line.rstrip().split()[1])
                    assert dim == _dim, '{} != {}'.format(_dim, dim)
                    first_line = False
                    continue

            line = line.rstrip()
            if not line:
                if error == 'warn':
                    print(line)
                continue

            tokens = line.rsplit(sep, dim)

            v = [float(x) for x in tokens[1:]]
            if len(v) != dim:
                continue

            vectors.append(v)
            words.append(tokens[0])

            assert len(vectors[-1]) == dim, line
            if chunk_size and chunk_size > 0 and len(vectors) == chunk_size:
                yield np.array(vectors, dtype=np.float32), words
                del vectors[:]
                del words[:]

            lines_num += 1
            first_line = False if first_line else False

    if len(vectors) > 0:
        yield np.array(vectors, dtype=np.float32), words


def unseen_questions(analogy, unseen_words):
    count, unseen_count = 0, 0
    for analogy_type in analogy:
        for a,b,c,d in analogy[analogy_type]["questions"]:
            count += 1
            unseen_count += a in unseen_words or b in unseen_words or c in unseen_words or d in unseen_words
    return unseen_count, unseen_count/count

@time_cost
@print_func
def read_analogy(path, iw=None):
    '''
        读取相似问题库
    :param path:
    :param iw:
    :return:
    '''
    new_iw = False
    if iw is None:
        iw = []
        new_iw = True

    analogy = {}
    analogy_type = ""
    with open(path) as f:
        for line in tqdm(f):
            if line.strip().split()[0] == ':':
                analogy_type = line.strip().split()[1]
                analogy[analogy_type] = {}
                analogy[analogy_type]["questions"] = []
                analogy[analogy_type]["total"] = 0
                analogy[analogy_type]["seen"] = 0
                continue
            analogy_question = line.strip().split()
            for w in analogy_question[:3]:
                if new_iw:
                    if not w in iw:
                        iw.append(w)
                else:
                    if w not in iw: # 词不在embedding里面
                        continue

            analogy[analogy_type]["total"] += 1
            analogy[analogy_type]["questions"].append(analogy_question)

        return analogy, iw if new_iw else None

@print_func
def normalize(matrix):
    norm = np.sqrt(np.sum(matrix * matrix, axis=1))
    matrix = matrix / norm[:, np.newaxis]
    return matrix

def guess(sims, matrix_iw, wi, word_a, word_b, word_c):
    '''

    :param sims: [m, V] 相似矩阵
    :param analogy:
    :param analogy_type:
    :param iw:
    :param wi:
    :param word_a:
    :param word_b:
    :param word_c:
    :return:
    '''
    matrix_wi = {w:i for i,w in enumerate(matrix_iw)}
    sim_a = sims[wi[word_a]]
    sim_b = sims[wi[word_b]]
    sim_c = sims[wi[word_c]]

    add_sim = -sim_a+sim_b+sim_c
    if word_a in matrix_wi:
        add_sim[matrix_wi[word_a]] = 0
    if word_b in matrix_wi:
        add_sim[matrix_wi[word_b]] = 0
    if word_c in matrix_wi:
        add_sim[matrix_wi[word_c]] = 0
    idx = np.nanargmax(add_sim)
    guess_add = matrix_iw[idx]
    add_max = add_sim[idx]

    mul_sim = sim_b * sim_c * np.reciprocal(sim_a+0.01)
    if word_a in matrix_wi:
        mul_sim[matrix_wi[word_a]] = 0
    if word_b in matrix_wi:
        mul_sim[matrix_wi[word_b]] = 0
    if word_c in matrix_wi:
        mul_sim[matrix_wi[word_c]] = 0
    idx = np.nanargmax(mul_sim)
    guess_mul = matrix_iw[idx]
    mul_max = mul_sim[idx]
    return (guess_add, add_max), (guess_mul, mul_max)

def find_sim(analogy, analogy_matrix, matrix, wi, matrix_iw, result=None, oov_words=None):
    if result is None:
        result = dict()

    for analogy_type in tqdm(analogy.keys(), desc='find_sim: '):  # Calculate the accuracy for each relation type
        result.setdefault(analogy_type, defaultdict(list))
        sims = analogy_matrix.dot(matrix.T).astype(np.float32)
        # Transform similarity scores to positive numbers (for mul evaluation)
        sims += 1
        sims /= 2
        for question in analogy[analogy_type]["questions"]:  # Loop for each analogy question
            word_a, word_b, word_c, word_d = question
            if not oov_words or not (word_a in oov_words or word_b in oov_words or word_c in oov_words): # 任何一个oov，则不计算
                (guess_add, add_max), (guess_mul, mul_max) = guess(sims, matrix_iw, wi, word_a, word_b, word_c)
                result[analogy_type][word_d].append([(guess_add, add_max), (guess_mul, mul_max)])
    return result

@time_cost
@print_func
def agg_result(sim_dict, analogy):
    results = dict()
    for analogy_type in sim_dict.keys():  # Calculate the accuracy for each relation type
        correct_add_num, correct_mul_num = 0, 0
        for word_d, items in sim_dict[analogy_type].items():  # Loop for each analogy question
            guess_add = sorted(items, key = lambda x: x[0][1], reverse=True)[0][0][0]
            if guess_add == word_d:
                correct_add_num += 1

            guess_mul = sorted(items, key=lambda x: x[1][1], reverse=True)[0][1][0]
            if guess_mul == word_d:
                correct_mul_num += 1

        seen = len(sim_dict[analogy_type])
        cov = float(seen / analogy[analogy_type]["total"])
        if seen == 0:
            acc_add = 0
            acc_mul = 0
            print(analogy_type + " add/mul: " + str(round(0.0, 3)) + "/" + str(round(0.0, 3)))
        else: # 仅仅统计过存在a&b&c的记录
            acc_add = float(correct_add_num) / seen
            acc_mul = float(correct_mul_num) / seen
            print(analogy_type + " add/mul: " + str(round(acc_add, 3)) + "/" + str(round(acc_mul, 3)))
        # Store the results
        results[analogy_type] = {}
        results[analogy_type]["coverage"] = [cov, seen, analogy[analogy_type]["total"]]
        results[analogy_type]["accuracy_add"] = [acc_add, correct_add_num, seen]
        results[analogy_type]["accuracy_mul"] = [acc_mul, correct_mul_num, seen]

    correct_add_num, correct_mul_num, total_seen = 0, 0, 0
    for analogy_type in results:
        correct_add_num += results[analogy_type]["accuracy_add"][1]
        correct_mul_num += results[analogy_type]["accuracy_mul"][1]
        total_seen += results[analogy_type]["coverage"][1]

    # print results
    if total_seen == 0:
        print("Total accuracy (add): {}".format(str(round(0.0, 3))), 'zero cover')
        print("Total accuracy (mul): {}".format(str(round(0.0, 3))), 'zero cover')
    else:
        print("Total accuracy (add): " + str(round(float(correct_add_num)/total_seen, 3)))
        print("Total accuracy (mul): " + str(round(float(correct_mul_num)/total_seen, 3)))

@time_cost
@print_func
def eval():
    myParser = argparse.ArgumentParser()
    myParser.add_argument('-v', '--vectors', type=str, default="embedding_sample/dense_small.txt", help="Vectors path")
    myParser.add_argument('-a', '--analogy', type=str, default="CA8/morphological.txt", help="Analogy benchmark path")
    myParser.add_argument('-d', '--dim', type=int, default=0, help="vector dim")
    myParser.add_argument('-s', '--sep', type=str, default='\t', help="sep")
    myParser.add_argument('-c', '--chunk-size', type=int, default=0, help="load word vectors by row chunk size")
    myParser.add_argument('--header', default=False, action='store_true', help="file with header")
    args = myParser.parse_args()

    analogy, analogy_iw = read_analogy(args.analogy, iw = None)  # Read analogy questions
    iw, wi, dim, analogy_matrix, unseen_words = build_vocab(args.vectors, iw=analogy_iw, header=args.header, dim=args.dim, sep=args.sep, build_all_vocab=False)
    unseen_count, unseen_ratio = unseen_questions(analogy, unseen_words=unseen_words)
    print('unseen_words: {}, unseen_count: {}, unseen_ratio: {}'.format(len(unseen_words), unseen_count, unseen_ratio))
    matrix_iter = read_vectors(args.vectors, chunk_size=args.chunk_size, header=args.header, dim=args.dim, sep=args.sep)

    result = defaultdict(list)
    for matrix, matrix_iw in tqdm(matrix_iter, desc='matrix chunk'):
        matrix = normalize(matrix)
        find_sim(analogy, analogy_matrix, matrix, wi, matrix_iw, result=result, oov_words=unseen_words)
    agg_result(result, analogy)


if __name__ == '__main__':
    eval()