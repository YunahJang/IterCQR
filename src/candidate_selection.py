import math
import torch
import os
import numpy as np
import json
from tqdm import tqdm
import copy
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,GPT2TokenizerFast
import argparse
from transformers import pipeline


parser = argparse.ArgumentParser()
parser.add_argument("--candidate_path", type=str, default="output/train_qrecc/Checkpoint/KD-ANCE-prefix-oracle-0.5-best-model")
parser.add_argument('--output_path', type=str, default="output/qrecc/QR/test_QRIR_oracle_prefix.json")
parser.add_argument('--selection_type', type=str, default="QA") #QA
args = parser.parse_args()  

#OUTPUT_FILE_PATH=args.output_path
#OUTPUT_FILE_PATH="/itercqr/data/datasets/topiocqa/train_new_with_bestcandidates_loss_answer_history_GL_iter1.json"
OUTPUT_FILE_PATH="/itercqr/data/datasets/topiocqa/train_new_with_bestcandidates_qa_model.json"
#INPUT_FILE_PATH=args.candidate_path
INPUT_FILE_PATH="/itercqr/data/datasets/topiocqa/train_new_with_candidates.json"
ORIGINAL_FILE_PATH="/itercqr/data/datasets/topiocqa/train_new.json"

with open(ORIGINAL_FILE_PATH, 'r') as f:
    data=[json.loads(l) for l in f.readlines()]

with open(INPUT_FILE_PATH, 'r') as f:
    input=[json.loads(l) for l in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')


model.to(device)

new_data=copy.deepcopy(data)

print(INPUT_FILE_PATH)
print(OUTPUT_FILE_PATH)
if args.selection_type=="GL": 
    with open(OUTPUT_FILE_PATH, 'w') as f:
        for i in tqdm(range(len(data))):
            ctx_utts_text = ""
            curr_query= data[i]["query"]
            history_query = data[i]['history_query']
            history_answer = data[i]['history_answer']
            curr_answer = data[i]['answer']
            score=[]
            for k in range(len(history_query)):
                ctx_utts_text+=" "
                ctx_utts_text+=history_query[k]
                ctx_utts_text+=" "
                ctx_utts_text+=history_answer[k]
            #for j in range(len(input[i]["model_generated_query_candidates"])):
            for j in range(len(input[i]["oracle_utt_text"])):    
                candidate = input[i]["oracle_utt_text"][j]
                if ctx_utts_text==[]:
                    break
                else: 
                    input_text="Context: "+ctx_utts_text+"Question: "+candidate+"Answer: "+curr_answer
                    #input_text="Question: "+candidate+"Answer: "+curr_answer
                    input_ids = torch.tensor(tokenizer.encode(input_text))
                    output = model(input_ids.to(device), labels=input_ids.to(device))
                    score.append(output.loss.item())
            best_candidate_index=np.argmin(score)
            best_candidate=input[i]["oracle_utt_text"][best_candidate_index]
            new_data[i]["best_candidate"]=best_candidate
            new_data[i]["score"]=score
            f.write(json.dumps(new_data[i]) + '\n')      

elif args.selection_type=="QA": 
    model_checkpoint = "consciousAI/question-answering-roberta-base-s-v2"
    question_answerer = pipeline("question-answering", model=model_checkpoint, device=device)
    with open(OUTPUT_FILE_PATH, 'w') as f:
        for i in tqdm(range(len(data))):
            ctx_utts_text = ""
            curr_query= data[i]["query"]
            curr_answer = data[i]['answer']
            gold_passage= data[i]['pos_docs'][0]
            score=[]
            candidates=input[i]["model_generated_query_candidates"]
            for j in range(len(candidates)):
                candidate = candidates[j]
                if gold_passage==[]:
                    break
                else: 
                    output=question_answerer(question=candidate, context=gold_passage)
                    score.append([output['score'], candidate, output['answer']])
            score_cand_sorted = sorted(score, key=lambda x: x[0], reverse=True)
            new_data[i]["best_candidate"]=score_cand_sorted[0][1]
            new_data[i]["model_generated_query_candidates"]=score_cand_sorted
            f.write(json.dumps(new_data[i]) + '\n')    