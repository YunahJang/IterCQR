from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import pdb
import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
import toml
import os
import math

from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tensorboardX import SummaryWriter
from data_structure_yunah import T5RewriterIRDataset_noanswer_qrecc, T5RewriterIRDataset_noanswer_topiocqa, T5RewriterIRDataset_qrecc, T5RewriterIRDataset_simq_qrecc, T5RewriterIRDataset_simq_topiocqa, T5RewriterIRDataset_topiocqa
from data_structure_yunah_mbr import T5RewriterIRDataset_topiocqa_mbr, T5RewriterIRDataset_qrecc_mbr
from src.models import load_model
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer, print_res
#from data_structure_yunah import T5RewriterIRDataset_qrecc, T5RewriterIRDataset_topiocqa,T5RewriterIRDataset_simq_qrecc, T5RewriterIRDataset_simq_topiocqa, T5RewriterIRDataset_noanswer_qrecc, T5RewriterIRDataset_noanswer_topiocqa, T5RewriterIRDataset_gpt_history_qrecc



def save_model(args, model, query_tokenizer, save_model_order, epoch, step, loss):
    output_dir = oj(args.model_output_path, '{}-{}'.format("KD-ANCE-prefix", args.decode_type), f'{epoch}')
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    #model_to_save.t5.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_kd_loss(query_embs, kd_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, kd_embs)


def train(args, log_writer):
    passage_tokenizer, passage_encoder = load_model("ANCE_Passage", args.pretrained_passage_encoder)
    passage_encoder = passage_encoder.to(args.device)
   
    query_tokenizer = T5Tokenizer.from_pretrained(args.pretrained_query_encoder_tokenizer)
    query_encoder = T5ForConditionalGeneration.from_pretrained(args.pretrained_query_encoder).to(args.device)

    
    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # data prepare
    if args.train_dataset=="qrecc":
        train_dataset = T5RewriterIRDataset_qrecc(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="topiocqa":
        train_dataset = T5RewriterIRDataset_topiocqa(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="topiocqa_simq": 
        train_dataset = T5RewriterIRDataset_simq_topiocqa(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="qrecc_simq":
        train_dataset = T5RewriterIRDataset_simq_qrecc(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="qrecc_noanswer":
        train_dataset = T5RewriterIRDataset_noanswer_qrecc(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="topiocqa_noanswer":
        train_dataset = T5RewriterIRDataset_noanswer_topiocqa(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="qrecc_gpt":
        train_dataset = T5RewriterIRDataset_qrecc(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="topiocqa_iterative":
        train_dataset = T5RewriterIRDataset_topiocqa(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="qrecc_iterative":
        train_dataset = T5RewriterIRDataset_qrecc(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="topiocqa_gpt":
        train_dataset = T5RewriterIRDataset_topiocqa(args, query_tokenizer, args.train_file_path)
    elif args.train_dataset=="topiocqa_mbr":
        train_dataset = T5RewriterIRDataset_topiocqa_mbr(args, query_tokenizer, args.train_file_path) 
    elif args.train_dataset=="topiocqa_hardmbr":
        train_dataset = T5RewriterIRDataset_topiocqa_mbr(args, query_tokenizer, args.train_file_path) 
    elif args.train_dataset=="topiocqa_reinforce":
        train_dataset = T5RewriterIRDataset_topiocqa_mbr(args, query_tokenizer, args.train_file_path)    
    elif args.train_dataset=="qrecc_mbr":
        train_dataset = T5RewriterIRDataset_qrecc_mbr(args, query_tokenizer, args.train_file_path)    
    # train_dataset = T5RewriterIRDataset_topiocqa(args, query_tokenizer, args.train_file_path)
    # train_dataset = T5RewriterIRDataset_qrecc(args, query_tokenizer, args.train_file_path)
    train_loader = DataLoader(train_dataset, 
                                #sampler=train_sampler,
                                batch_size = args.batch_size, 
                                shuffle=True, 
                                collate_fn=train_dataset.get_collate_fn(args))

    logger.info("train samples num = {}".format(len(train_dataset)))
    
    total_training_steps = args.num_train_epochs * (len(train_dataset) /args.gradient_accumulation_steps // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0
    save_model_order = 0

    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)
    logsoftmax=torch.nn.LogSoftmax(dim=-1)
    softmax=torch.nn.Softmax(dim=-1)
    best_loss = 1000
    for epoch in epoch_iterator:
        query_encoder.train()
        query_encoder.zero_grad()
        passage_encoder.eval()
        i=-1
        for batch in tqdm(train_loader,  desc="Step"):
            i+=1
            # query_encoder.zero_grad()

            bt_conv_query = batch['bt_input_ids'].to(args.device) # B * len
            bt_conv_query_mask = batch['bt_attention_mask'].to(args.device)
            bt_pos_docs = batch['bt_pos_docs'].to(args.device) # B * len one pos
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            bt_neg_docs = batch['bt_neg_docs'].to(args.device) # B * len batch size negs
            bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)
            bt_oracle_query = batch['bt_labels'].to(args.device) # B * num_seqs * len
            
            with torch.no_grad():
                # freeze passage encoder's parameters
                pos_doc_embs = passage_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                #neg_doc_embs = passage_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # B * dim, hard negative

            num_seqs = bt_oracle_query.size(1)
            if 'bt_cossim_score' in batch or args.train_dataset=="topiocqa_mbr" or args.train_dataset=="topiocqa_hardmbr" or args.train_dataset=="qrecc_mbr":
                decode_loss=0
                ranking_loss=0
                for bid in range(bt_oracle_query.size(0)):
                    bt_conv_query_ = bt_conv_query[bid] # len
                    bt_conv_query_ = bt_conv_query_.repeat(num_seqs, 1) # num_seq * len

                    bt_conv_query_mask_ = bt_conv_query_mask[bid] # len
                    bt_conv_query_mask_ = bt_conv_query_mask_.repeat(num_seqs, 1) # num_seq * len

                    bt_oracle_query_ = bt_oracle_query[bid] # num_seq * len

                    # the model processes 10 seqs at a time regardless of the batch size
                    output = query_encoder(input_ids=bt_conv_query_, 
                            attention_mask=bt_conv_query_mask_, 
                            labels=bt_oracle_query_)
                    #pdb.set_trace()
                    conv_query_embs = output.encoder_last_hidden_state[:, 0]
                    bt_cossim_score=batch['bt_cossim_score'][bid].to(args.device) # num_seqs
                    bt_prob_score=batch['bt_prob_score'][bid].to(args.device) # num_seqs
                    
                    logit_logsoftmax=logsoftmax(output.logits) # num_seqs * len * vocab_size
                    candidate_log_prob=[]
                    for j in range(num_seqs):
                        candidate_loss=0
                        if torch.where(bt_oracle_query_[j]==-100)[0].nelement()==0:
                            st_index=len(bt_oracle_query_[j])
                        else: 
                            st_index=torch.where(bt_oracle_query_[j]==-100)[0][0].item()
                        for k in range(st_index):
                            candidate_loss+=logit_logsoftmax[j][k][bt_oracle_query_[j][k]]
                        candidate_log_prob.append(candidate_loss)

                    candidate_log_prob=torch.stack(candidate_log_prob, dim=0)
                    normalized_candidate_prob=softmax(candidate_log_prob) # num_seqs
                    normalized_candidate_log_prob=logsoftmax(candidate_log_prob) # num_seqs
                    prob_score=softmax(bt_prob_score)

                    if args.train_dataset=="topiocqa_hardmbr":
                        decode_loss+=normalized_candidate_prob[0]

                    elif args.train_dataset=="topiocqa_reinforce":
                        normalized_candidate_prob_detached = normalized_candidate_prob.detach()

                        for m in range(num_seqs):
                            decode_loss+= normalized_candidate_log_prob[m] * bt_cossim_score[m] * normalized_candidate_prob_detached[m] / prob_score[m]

                    else: 
                        for m in range(num_seqs):
                            decode_loss+=normalized_candidate_prob[m] * bt_cossim_score[m] 

                    ranking_loss+=cal_kd_loss(conv_query_embs, pos_doc_embs[bid].repeat(num_seqs, 1))

                decode_loss /= bt_oracle_query.size(0)
                ranking_loss /= bt_oracle_query.size(0)
            else:
                output = query_encoder(input_ids=bt_conv_query, 
                         attention_mask=bt_conv_query_mask, 
                         labels=bt_oracle_query)
                decode_loss=output.loss # B * dim
            conv_query_embs = output.encoder_last_hidden_state[:, 0]


            if args.train_dataset=="topiocqa_mbr" or args.train_dataset=="topiocqa_reinforce" or args.train_dataset=="qrecc_mbr" or args.train_dataset=="topiocqa_hardmbr":
                loss=-decode_loss
            else: 
                loss=decode_loss
            if args.kd_loss=="true":
            #ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
                ranking_loss = cal_kd_loss(conv_query_embs, pos_doc_embs)
                loss = decode_loss + args.alpha * ranking_loss
            #loss_=loss.clone()
            (loss/args.gradient_accumulation_steps).backward()
            #loss.backward()
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            if (i+1) % args.gradient_accumulation_steps ==0:
                optimizer.step()
                scheduler.step()
                query_encoder.zero_grad()
                #zero grad query encoder한테 하는거랑 같은건가
                #optimizer.zero_grad()

            # if math.isnan(loss.item()):
            #     pdb.set_trace()

            if args.print_steps > 0 and global_step % args.print_steps == 0:
                if args.kd_loss=="false":
                    logger.info("Epoch = {}, Global Step = {}, decode loss = {}, total loss = {}".format(
                    epoch + 1,
                    global_step,
                    decode_loss.item(),
                    loss.item()))
                else: 
                    logger.info("Epoch = {}, Global Step = {}, ranking loss = {}, decode loss = {}, total loss = {}".format(
                                    epoch + 1,
                                    global_step,
                                    ranking_loss.item(),
                                    decode_loss.item(),
                                    loss.item()))

            #log_writer.add_scalar("train_ranking_loss, decode_loss, total_loss", ranking_loss, decode_loss, loss, global_step)
            

            global_step += 1    # avoid saving the model of the first step.
            # save model finally
            if best_loss > loss:
                save_model(args, query_encoder, query_tokenizer, save_model_order, f'best', global_step, loss.item())
                best_loss = loss
                if args.kd_loss=="false":
                    logger.info("Epoch = {}, Global Step = {}, decode loss = {}, total loss = {}".format(
                    epoch + 1,
                    global_step,
                    decode_loss.item(),
                    loss.item()))
                else: 
                    logger.info("Epoch = {}, Global Step = {}, ranking loss = {}, decode loss = {}, total loss = {}".format(
                                    epoch + 1,
                                    global_step,
                                    ranking_loss.item(),
                                    decode_loss.item(),
                                    loss.item()))
        save_model(args, query_encoder, query_tokenizer, save_model_order, epoch+1, global_step, loss.item())

        # resample examples (shuffle simq list)
        if (args.decode_type == "sim_q") and (epoch != args.num_train_epochs - 1):
            if args.train_dataset=="topiocqa_simq": 
                train_dataset = T5RewriterIRDataset_simq_topiocqa(args, query_tokenizer, args.train_file_path)

                train_loader = DataLoader(train_dataset, 
                                    #sampler=train_sampler,
                                    batch_size = args.batch_size, 
                                    shuffle=True, 
                                    collate_fn=train_dataset.get_collate_fn(args))
                                    
            elif args.train_dataset=="qrecc_simq":
                train_dataset = T5RewriterIRDataset_simq_qrecc(args, query_tokenizer, args.train_file_path)
            
                train_loader = DataLoader(train_dataset, 
                                    #sampler=train_sampler,
                                    batch_size = args.batch_size, 
                                    shuffle=True, 
                                    collate_fn=train_dataset.get_collate_fn(args))

            elif args.train_dataset=="topiocqa": 
                train_dataset = T5RewriterIRDataset_topiocqa(args, query_tokenizer, args.train_file_path)

                train_loader = DataLoader(train_dataset, 
                                    #sampler=train_sampler,
                                    batch_size = args.batch_size, 
                                    shuffle=True, 
                                    collate_fn=train_dataset.get_collate_fn(args))

            elif args.train_dataset=="qrecc":
                train_dataset = T5RewriterIRDataset_qrecc(args, query_tokenizer, args.train_file_path)
            
                train_loader = DataLoader(train_dataset, 
                                    #sampler=train_sampler,
                                    batch_size = args.batch_size, 
                                    shuffle=True, 
                                    collate_fn=train_dataset.get_collate_fn(args))   
    logger.info("Training finish!")          
         


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_query_encoder_tokenizer", type=str, default="checkpoints/T5-base")
    parser.add_argument("--pretrained_query_encoder", type=str, default="checkpoints/T5-base")
    parser.add_argument("--pretrained_passage_encoder", type=str, default="checkpoints/ad-hoc-ance-msmarco")

    parser.add_argument("--train_file_path", type=str, default="datasets/qrecc/new_preprocessed/train_with_doc.json")
    parser.add_argument("--train_dataset", type=str, default="qrecc_simq")
    parser.add_argument("--num_candidates_for_training", type=int, default=1)
    parser.add_argument("--log_dir_path", type=str, default="output/train_topiocqa/Log")
    parser.add_argument('--model_output_path', type=str, default="output/train_topiocqa/Checkpoint")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train")
    parser.add_argument("--decode_type", type=str, default="oracle")
    parser.add_argument("--use_prefix", type=bool, default=True)

    parser.add_argument("--per_gpu_train_batch_size", type=int,  default=8)
    parser.add_argument("--use_data_percent", type=float, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10, help="graident accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=15, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--kd_loss", type=str, default="false")
    parser.add_argument("--print_steps", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
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
    log_writer = SummaryWriter(log_dir = args.log_dir_path)
    train(args, log_writer)
    log_writer.close()

