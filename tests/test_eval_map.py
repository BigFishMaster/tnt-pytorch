# coding=utf-8

import numpy as np
import os
import sys
import threading
import time

CALC_AP_TOOL = '/share/ananke/feature_train_eval/eval/eval_tool/calc_map'


def compute_euclidean_distance(Q, feats, names):
    dists = ((Q - feats) ** 2).sum(axis=1)
    idx = np.argsort(dists)
    dists = dists[idx]
    rank_names = [names[k] for k in idx]
    return idx, dists, rank_names


def compute_cosine_distance(Q, feats, names):
    """
    feats and Q: L2-normalize, n*d
    """
    dists = np.dot(Q, feats.T)
    #     print("dists:",dists)
    #     exit(1)
    idxs = np.argsort(dists)[::-1]
    rank_dists = dists[idxs]
    rank_names = [names[k] for k in idxs]
    return (idxs, rank_dists, rank_names)


def load_files(prefix, dim=512):
    feats = np.fromfile(prefix + ".fea", dtype=float, sep=" ")
    feats = feats.reshape(-1, dim)

    names = [n.strip() for n in open(prefix + ".txt", "r", encoding="utf8").readlines()]
    return feats, names


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def get_query_class(query_name):
    # set_1 = '89.data_arrangement_color_0121_all_sort'
    set_1 = '89.data_arrangement_color_0310'
    idx = query_name.find(set_1)
    if idx >= 0:
        cls_idx = idx + len(set_1) + 1
        return int(query_name[cls_idx: cls_idx + 3])
    return -1


def computer_ap(gt_prefix, ranked_list):
    with open(gt_prefix + '_good.txt', 'r') as fp:
        result = fp.readlines()

    gt = {r.strip() for r in result}

    pos_cnt = len(gt)
    # print ("pos_cnt:",pos_cnt)
    ap = 0.0
    ap_10 = 0.0
    ap_20 = 0.0
    ap_50 = 0.0
    p2 = -1
    p5 = -1
    p10 = -1
    ap2 = 0.0
    ap2_10 = 0.0
    ap2_20 = 0.0
    ap2_50 = 0.0

    old_recall = 0.0
    old_precision = 1.0
    intersect_size = 0.0
    total_precision = 0.0  # map2
    for i in range(len(ranked_list)):
        if 'all_eval' in ranked_list[i]:
            ranked_list[i] = ranked_list[i].replace('all_eval', 'baseset')

        if ranked_list[i] in gt:
            intersect_size += 1
            total_precision += intersect_size / (i + 1)

        recall = intersect_size / pos_cnt
        precision = intersect_size / (i + 1)
        ap += (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        if i + 1 == 10:
            ap_10 = ap
            ap2_10 = total_precision / float(intersect_size + 1e-8)
        if i + 1 == 20:
            ap_20 = ap
            ap2_20 = total_precision / float(intersect_size + 1e-8)
        if i + 1 == 50:
            ap_50 = ap
            ap2_50 = total_precision / float(intersect_size + 1e-8)

        if i + 1 == 2:
            p2 = precision
        elif i + 1 == 5 and pos_cnt >= 5:
            p5 = precision
        elif i + 1 == 10 and pos_cnt >= 10:
            p10 = precision

    ap2 = total_precision / float(intersect_size + 1e-8)
    return ap, p2, p5, p10, ap2, ap_10, ap_20, ap_50, ap2_10, ap2_20, ap2_50


def clac_map_adv(thread_id,
                 st, ed,
                 query_feats, query_names,
                 base_feats, base_names):
    print('Thread_adv %d begin, to process cnt %d' % (thread_id, ed - st))
    st_time = time.time()

    clac_result = []
    proc_cnt = 0
    i = st
    num = 100
    map_n = 100
    while i < ed - num:
        Q_mat = query_feats[i:i + num]
        dists = np.inner(Q_mat, base_feats)
        #         print("dists:",dists)
        idxs = np.argsort(dists)[:, ::-1]

        for q in range(num):
            if dists[q][idxs[q][0]] >= 0.999999:  # 最近邻就是查询自己
                rank_names = [base_names[k] for k in idxs[q][1:map_n + 1]]  # calc map@100
            else:
                rank_names = [base_names[k] for k in idxs[q][:map_n]]  # calc map@100

            gt_prefix = os.path.splitext(query_names[i + q])[0]

            ap, p2, p5, p10, ap2, ap_10, ap_20, ap_50, ap2_10, ap2_20, ap2_50 = computer_ap(gt_prefix, rank_names)
            cls = get_query_class(query_names[i + q])
            clac_result.append((cls, ap, p2, p5, p10, ap2, ap_10, ap_20, ap_50, ap2_10, ap2_20, ap2_50))

            proc_cnt += 1
            if proc_cnt % 50 == 0:
                print('TH-%d, processed image: %d' % (thread_id, proc_cnt))
                # print('Query %s, %s' % (query_names[i+q], ap))

        i += num

    if i < ed:
        Q_mat = query_feats[i:ed]
        dists = np.inner(Q_mat, base_feats)
        idxs = np.argsort(dists)[:, ::-1]

        for q in range(ed - i):
            if dists[q][idxs[q][0]] >= 0.999999:  # 最近邻就是查询自己
                rank_names = [base_names[k] for k in idxs[q][1:map_n + 1]]  # calc map@100
            else:
                rank_names = [base_names[k] for k in idxs[q][:map_n]]  # calc map@100

            # with open(rank_file, 'w') as f:
            #     f.write('\n'.join(rank_names))

            # compute mean average precision, precision@N
            gt_prefix = os.path.splitext(query_names[i + q])[0]

            # cmd = '%s %s %s' % (CALC_AP_TOOL, gt_prefix, rank_file)
            # ap = os.popen(cmd).read().strip()
            # cls = get_query_class(query_names[i])
            # clac_result.append('%d %s' % (cls, ap))

            ap, p2, p5, p10, ap2, ap_10, ap_20, ap_50, ap2_10, ap2_20, ap2_50 = computer_ap(gt_prefix, rank_names)
            cls = get_query_class(query_names[i + q])
            clac_result.append((cls, ap, p2, p5, p10, ap2, ap_10, ap_20, ap_50, ap2_10, ap2_20, ap2_50))

            proc_cnt += 1
            if proc_cnt % 50 == 0:
                print('TH-%d, processed image: %d' % (thread_id, proc_cnt))
                # print('Query %s, %s' % (query_names[i+q], ap))

    ed_time = time.time()

    print('Thread %d, totally cost %f s' % (thread_id, ed_time - st_time))

    return clac_result


def map_mt(do_QE, query_names, query_feats, base_names, base_feats, thnum):
    print("map_mt dist_type:", dist_type)
    query_cnt = len(query_names)
    print('Query cnt: %d' % query_cnt)

    each_thread_num = query_cnt // thnum
    mt = []
    for i in range(thnum):
        st = i * each_thread_num
        ed = st + each_thread_num
        if i == thnum - 1:
            ed = query_cnt

        t = MyThread(func=clac_map_adv, args=(do_QE, i, dist_type,
                                              st, ed,
                                              query_feats, query_names,
                                              base_feats, base_names))
        mt.append(t)
        t.start()

    all_result = []
    for t in mt:
        t.join()  # 一定要join，不然主线程比子线程跑的快，会拿不到结果
        result = t.get_result()
        all_result.extend(result)

    return all_result


def calc_each_class_result(result):
    all_map = []
    all_p2 = []
    all_p5 = []
    all_p10 = []
    all_map2 = []
    all_map_10 = []
    all_map_20 = []
    all_map_50 = []
    all_map2_10 = []
    all_map2_20 = []
    all_map2_50 = []
    for res in result:
        if res[1] >= 0:
            all_map.append(res[1])
        if res[2] >= 0:
            all_p2.append(res[2])
        if res[3] >= 0:
            all_p5.append(res[3])
        if res[4] >= 0:
            all_p10.append(res[4])
        if res[5] >= 0:
            all_map2.append(res[5])
        if res[6] >= 0:
            all_map_10.append(res[6])
        if res[7] >= 0:
            all_map_20.append(res[7])
        if res[8] >= 0:
            all_map_50.append(res[8])
        if res[9] >= 0:
            all_map2_10.append(res[9])
        if res[10] >= 0:
            all_map2_20.append(res[10])
        if res[11] >= 0:
            all_map2_50.append(res[11])

    mAP = sum(all_map) / (len(all_map) + 1e-8)
    p2 = sum(all_p2) / (len(all_p2) + 1e-8)
    p5 = sum(all_p5) / (len(all_p5) + 1e-8)
    p10 = sum(all_p10) / (len(all_p10) + 1e-8)
    mAP2 = sum(all_map2) / (len(all_map2) + 1e-8)
    mAP_10 = sum(all_map_10) / (len(all_map_10) + 1e-8)
    mAP_20 = sum(all_map_20) / (len(all_map_20) + 1e-8)
    mAP_50 = sum(all_map_50) / (len(all_map_50) + 1e-8)
    mAP2_10 = sum(all_map2_10) / (len(all_map2_10) + 1e-8)
    mAP2_20 = sum(all_map2_20) / (len(all_map2_20) + 1e-8)
    mAP2_50 = sum(all_map2_50) / (len(all_map2_50) + 1e-8)

    return mAP, p2, p5, p10, mAP2, mAP_10, mAP_20, mAP_50, mAP2_10, mAP2_20, mAP2_50


def evaluate(query_prefix, db_prefix, dist_type, thnum=1):
    # query expansion
    do_QE = False

    query_feats, query_names = load_files(query_prefix)
    db_feats, db_names = load_files(db_prefix)

    q_norm = np.linalg.norm(query_feats, axis=1, keepdims=True)
    query_feats = query_feats / q_norm

    db_feats = np.vstack([query_feats, db_feats])
    db_names = query_names + db_names

    db_norm = np.linalg.norm(db_feats, axis=1, keepdims=True)
    db_feats = db_feats / db_norm

    print("evaluate query_feats.shape:", query_feats.shape)
    print("evaluate db_feats.shape:", db_feats.shape)
    print("evaluate dist_type:", dist_type)

    time_st = time.time()
    result = map_mt(do_QE, query_names, query_feats, db_names, db_feats, thnum)
    time_ed = time.time()
    print(
        'Total time %f, each query %f ms' % (time_ed - time_st, 1000 * (time_ed - time_st) / (len(query_names) + 1e-8)))

    mAP, p2, p5, p10, mAP2, mAP_10, mAP_20, mAP_50, mAP2_10, mAP2_20, mAP2_50 = calc_each_class_result(result)

    result_name = "result_mAP.txt"

    f = open(os.path.join('result/', result_name), 'w')

    result = 'mAP@10, mAP@20, mAP@50, mAP@100, p@2, p@5, p@10, mAP2@10, mAP2@20, mAP2@50，mAP2@100\n \
              %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f' % (
    mAP_10, mAP_20, mAP_50, mAP, p2, p5, p10, mAP2_10, mAP2_20, mAP2_50, mAP2)
    print(result)
    f.write(result)
    f.close()


if __name__ == '__main__':
    print(sys.argv)

    query_prefix = sys.argv[1]
    db_prefix = sys.argv[2]
    dist_type = "cosine"
    evaluate(query_prefix, db_prefix, dist_type)
