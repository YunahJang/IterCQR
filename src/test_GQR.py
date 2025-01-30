import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import json
import argparse
import toml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from IPython import embed
import pdb
import torch
import math
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from utils import set_seed, format_nl_query
from data_structure import T5RewriterDataset_qrecc, T5RewriterDataset_topiocqa, T5RewriterDataset_cast, T5RewriterDataset_simq_qrecc, T5RewriterDataset_noanswer_qrecc, T5RewriterDataset_simq_topiocqa, T5RewriterDataset_gpt_history_qrecc
import numpy as np


def inference_t5qr(args):
    if args.model_type == "T5":
        tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint_path)
        model.to(args.device)
    elif args.model_type == "BART":
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint_path)
        model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint_path)
        model.to(args.device)

    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    if args.test_dataset=="qrecc":
        test_dataset = T5RewriterDataset_qrecc(args, tokenizer, args.test_file_path)
    elif args.test_dataset=="topiocqa":
        test_dataset = T5RewriterDataset_topiocqa(args, tokenizer, args.test_file_path)
    elif args.test_dataset=="topiocqa_simq":
        test_dataset = T5RewriterDataset_simq_topiocqa(args, tokenizer, args.test_file_path)
    elif args.test_dataset=="qrecc_simq":
        test_dataset = T5RewriterDataset_simq_qrecc(args, tokenizer, args.test_file_path, args.rewrite_file_path)
    elif args.test_dataset=="qrecc_noanswer":
        test_dataset = T5RewriterDataset_noanswer_qrecc(args, tokenizer, args.test_file_path)
    elif args.test_dataset=="qrecc_gpt":
        test_dataset = T5RewriterDataset_gpt_history_qrecc(args, tokenizer, args.test_file_path)
    elif args.test_dataset=="cast19":
        test_dataset = T5RewriterDataset_cast(args, tokenizer, args.test_file_path)
    elif args.test_dataset=="cast20":
        test_dataset = T5RewriterDataset_cast(args, tokenizer, args.test_file_path)
    

    args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #ddp_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, 
                                  shuffle=False,
                                  batch_size=args.batch_size, 
                                  collate_fn=test_dataset.get_collate_fn(args))
    
    # with open('/itercqr/data/datasets/topiocqa/train_new.json', 'r') as f:
    #     ori_data = [json.loads(x) for x in f.readlines()]
    #     global_i = 0
    # begin to inference
    with open(args.output_file_path, "w") as f:
        with torch.no_grad():
            model.eval()
            model.generation_config.length_penalty=0

            for batch in tqdm(test_dataloader, desc="Step"):
                bt_input_ids = batch["bt_input_ids"].to(args.device)
                bt_attention_mask = batch["bt_attention_mask"].to(args.device)
                if args.n_gpu > 1:
                    output_seqs = model.module.generate(input_ids=bt_input_ids, 
                                                        attention_mask=bt_attention_mask, 
                                                        do_sample=False,
                                                        max_length=args.max_query_length,
                                                        )
                    
                else:
                    if args.do_sample=='true':
                        output_seqs = model.generate(input_ids=bt_input_ids, 
                                                            attention_mask=bt_attention_mask, 
                                                            do_sample=True,
                                                            max_length=args.max_query_length,
                                                            top_k=args.top_k, top_p=args.top_p, 
                                                            num_return_sequences=args.num_return_sequences,
                                                            output_scores=True,
                                                            return_dict_in_generate=True
                                                            )
                        # pdb.set_trace()
                        # probabilities = output_seqs
                    else:
                        output_seqs = model.generate(input_ids=bt_input_ids, 
                                                            attention_mask=bt_attention_mask, 
                                                            do_sample=False,
                                                            max_length=args.max_query_length,
                                                            num_beams=args.num_beams,
                                                            num_return_sequences=args.num_return_sequences,
                                                            output_scores=True,
                                                            return_dict_in_generate=True
                                                            )
                        if args.num_return_sequences > 1:
                            transition_scores = model.compute_transition_scores(output_seqs.sequences, output_seqs.scores, output_seqs.beam_indices, normalize_logits=True)              
                            probabilities = torch.exp(transition_scores.sum(axis=1)).tolist()
                        
                # if args.num_return_sequences > 1:
                #     transition_scores = model.compute_transition_scores(output_seqs.sequences, output_seqs.scores, output_seqs.beam_indices, normalize_logits=True)              
                #     probabilities = torch.exp(transition_scores.sum(axis=1)).tolist()
                output_seqs = output_seqs['sequences']

                outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
                total_length=len(outputs)
                if args.num_return_sequences > 1:
                    outputs_ = []
                    for i in range(len(outputs)//args.num_return_sequences):
                        outputs_.append(outputs[i*args.num_return_sequences:(i+1)*args.num_return_sequences])
                    outputs = outputs_

                    if args.do_sample=='false':
                        probabilities_ = []
                        for i in range(total_length//args.num_return_sequences):
                            probabilities_.append(probabilities[i*args.num_return_sequences:(i+1)*args.num_return_sequences])
                        probabilities = probabilities_
                for i in range(len(outputs)):
                    record = {}
                    record["sample_id"] = batch["bt_sample_ids"][i]
                    if args.decode_type == "oracle":
                        record["oracle_utt_text"] = outputs[i]
                        if args.num_return_sequences > 1 and args.do_sample=='false':
                            record["probability"] = probabilities[i]
                    elif args.decode_type == "answer":
                        record["answer_utt_text"] = outputs[i]
                    elif args.decode_type == "sim_q":
                        record["similar_query"] = outputs[i]
                    elif args.decode_type == "next_q":
                        record["next_q_utt_text"] = outputs[i]
                    elif args.decode_type == "new_history":
                        record["new_history_query"] = outputs[i]
                    record["cur_utt_text"] = batch["bt_cur_utt_text"][i]

                    f.write(json.dumps(record) + '\n') 

    logger.info("Inference finsh!")
    

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_checkpoint_path", type=str, default="output/train_qrecc/Checkpoint/KD-ANCE-prefix-oracle-0.5-best-model")
    parser.add_argument("--test_file_path", type=str, default="datasets/qrecc/new_preprocessed/test.json")
    parser.add_argument('--output_file_path', type=str, default="output/qrecc/QR/test_QRIR_oracle_prefix.json")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_test")
    parser.add_argument("--decode_type", type=str, default="oracle")
    parser.add_argument("--model_type", type=str, default="T5")
    parser.add_argument("--use_last_response", type=bool, default=False)
    parser.add_argument("--use_prefix", type=bool, default=True)
    parser.add_argument("--test_dataset", type=str, default="qrecc")
    parser.add_argument("--rewrite_file_path", type=str, default="/itercqr/output/qrecc/rewrite/epoch15output.json")

    # sampling hyperparameters
    parser.add_argument("--do_sample", type=str, default='false')
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.8)

    parser.add_argument("--per_gpu_eval_batch_size", type=int,  default=32)
    parser.add_argument("--use_data_percent", type=float, default=1)
    
    parser.add_argument("--max_query_length", type=int, default=64, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()


    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#, args.local_rank)
    args.device = device


    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    inference_t5qr(args)
