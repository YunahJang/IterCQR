from re import T
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
from os import path
from os.path import join as oj
import toml
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

def check_dir_exist_or_build(dir_list, erase_dir_content = None):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
    if erase_dir_content:
        for dir_path in erase_dir_content:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)

def main():
    args = get_args()
    
    query_list = []
    qid_list = []
    print(args.input_query_path)
    with open(args.input_query_path, "r") as f:
        data = f.readlines()

    if args.input_query_path_2 is not None:
        with open(args.input_query_path_2, "r") as f2:
            data_2 = f2.readlines()
    # with open(args.input_query_path_3, "r") as f2:
    #     data_3 = f2.readlines()

    n = len(data)
    # if args.only_250==True:
    #     n=250
    for i in range(n):
        data[i] = json.loads(data[i])

        if args.query_type == "raw":
            query = data[i]["query"]
        elif args.query_type == "rewrite":
            query = data[i]['output']
        elif args.query_type == "oracle":
            #query = data[i]['oracle_query']
            query = data[i]['oracle_utt_text']
            if "|||" in query:
                query = query.replace('|||', ' ')
        elif args.query_type == "decode":
            query = data[i]['oracle_utt_text']
            if "|||" in query:
                query = query.replace('|||', ' ')
            if args.eval_type == "answer":
                data_2[i] = json.loads(data_2[i])
                query = data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+answer":
                data_2[i] = json.loads(data_2[i])
                query = query + ' ' + data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+history":
                data_2[i] = json.loads(data_2[i])
                query = query + ' ' + data_2[i]['new_history_query']
            elif args.eval_type == "history":
                data_2[i] = json.loads(data_2[i])
                query = data_2[i]['new_history_query']
            elif args.eval_type == "rewrite+history+answer":
                data_2[i] = json.loads(data_2[i])
                data_3[i] = json.loads(data_3[i])
                query = query + ' ' + data_2[i]['new_history_query'] + ' ' +data_3[i]['answer_utt_text']
            elif args.eval_type == "llm_rewrite":
                query = data[i]['model_rewrite']
            elif args.eval_type == "history_only":

                cur_utt_text = data[i]['cur_utt_text']
                ctx_utts_text = data[i]['ctx_utts_text']

                if args.use_prefix:
                    cur_utt_text = "question: " + cur_utt_text
                    first_context = True

                flat_concat=cur_utt_text
                for j in range(len(ctx_utts_text) - 1, -1, -1):
                    if args.use_prefix and first_context:
                        ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                        first_context = False
                    utt = ctx_utts_text[j]
                    flat_concat+=' '
                    flat_concat+=utt
                
                query=flat_concat
            
        query_list.append(query)
        qid_list.append(data[i]['sample_id'])
    
    # import pdb; pdb.set_trace()
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    print('search begins')
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 40)
    print("open file!!!!!")
    
    with open(oj(args.output_dir_path, args.trec_file_name), "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid[3:],
                                                i+1,
                                                -i - 1 + 200,
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')


    res = print_res(oj(args.output_dir_path, args.trec_file_name), args.gold_qrel_file_path, args.rel_threshold)
    print("res!")
    print(res)
    if args.result_save_path is not None:
        with open(args.result_save_path, 'a') as f:
            f.write('QRECC BM25' + '\n')
            f.write(str(res) + '\n\n')
    return res


def print_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            "MAP": np.average(map_list),
            "MRR": np.average(mrr_list),
            "NDCG@3": np.average(ndcg_3_list), 
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list),
        }

    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_query_path", type=str, default="/itercqr/output/topiocqa/iterative/qrecc_cossim_iter3_epoch5outputs.json")
    #parser.add_argument("--input_query_path", type=str, default="/itercqr/output/gpt/rewrite_input.json")
    parser.add_argument("--input_query_path_2", type=str, default="/itercqr/output/qrecc/gpt/historyinput_epoch15output.json")
    parser.add_argument("--input_query_path_3", type=str, default="/itercqr/output/qrecc/expansion/epoch15output.json")
    parser.add_argument("--index_dir_path", type=str, default="/itercqr/data/bm25_index/indexes_qrecc")
    parser.add_argument("--output_dir_path", type=str, default="../output/qrecc/bm25")
    parser.add_argument("--gold_qrel_file_path", type=str, default="../data/new_preprocessed/qrecc_qrel.tsv")
    parser.add_argument("--trec_file_name", type=str, default="bm25_history_only_qrecc_test.trec")
    parser.add_argument("--query_type", type=str, default="oracle")
    parser.add_argument("--eval_type", type=str, default="history_only")
    parser.add_argument("--bm25_k1", type=float, default="0.82")
    parser.add_argument("--bm25_b", type=float, default="0.68")
    parser.add_argument("--use_prefix", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default="100")
    parser.add_argument("--rel_threshold", type=int, default="1")
    parser.add_argument("--only_250", type=bool, default=False)
    parser.add_argument("--result_save_path", type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
