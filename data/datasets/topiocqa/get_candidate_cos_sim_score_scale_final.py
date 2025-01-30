import json
import torch
from transformers import PreTrainedModel, RobertaConfig, RobertaModel, RobertaTokenizer
from tqdm import tqdm
import copy
import pickle
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--candidate_path", type=str, default="output/train_qrecc/Checkpoint/KD-ANCE-prefix-oracle-0.5-best-model")
parser.add_argument('--output_path', type=str, default="output/qrecc/QR/test_QRIR_oracle_prefix.json")
parser.add_argument('--get_probs', type=str, default="true")
parser.add_argument('--scaling_type', type=str, default=None)
args = parser.parse_args()    

#OUTPUT_PATH = 'train_new_with_candidates_cos_sim_for_iter2.json'
OUTPUT_PATH=args.output_path
DATA_PATH='/itercqr/data/datasets/topiocqa/train_new.json'
#CAND_PATH='/itercqr/data/datasets/topiocqa/iterative/cossim_train_candidates_from_iter1_ckpt_for_iter2.json'
CAND_PATH=args.candidate_path

class AnceEncoder(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'ance_encoder'
    load_tf_weights = None
    # _keys_to_ignore_on_load_missing = [r'position_ids']
    # _keys_to_ignore_on_load_unexpected = [r'pooler', r'classifier']

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        self.embeddingHead = torch.nn.Linear(config.hidden_size, 768)
        self.norm = torch.nn.LayerNorm(768)
        self.init_weights()

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.roberta.init_weights()
        self.embeddingHead.apply(self._init_weights)
        self.norm.apply(self._init_weights)

    def forward(
            self,
            input_ids,
            attention_mask,
    ):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.roberta.config.pad_token_id)
            )
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.norm(self.embeddingHead(pooled_output))
        return pooled_output


model_path = "castorini/ance-msmarco-passage"
config = RobertaConfig.from_pretrained(
    model_path,
    finetuning_task="MSMarco",
)
tokenizer = RobertaTokenizer.from_pretrained(
    model_path,
    do_lower_case=True
)

model = AnceEncoder.from_pretrained(model_path, config=config).cuda()
model.eval()

with open(CAND_PATH, 'r') as f:
    data = [json.loads(x) for x in f.readlines()]

with open(DATA_PATH, 'r') as f:
    orig_data = [json.loads(x) for x in f.readlines()]
    
embs = torch.load('/itercqr/data/datasets/topiocqa/train_new_pos_docs_emb.pt')
min=1
max=0
out = []
score_list=[]
with open(OUTPUT_PATH, 'w') as fw:
    for i in tqdm(range(len(data))):
        item=orig_data[i]
        cand=data[i]
        example_id = item['id']
        candidates = cand['oracle_utt_text']
        if args.get_probs=='true':
            probabilities = cand['probability']
        #candidates = item['model_generated_query_candidates']
        pos_docs_emb = embs[example_id].cuda()

        inputs = tokenizer(candidates, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        candidates_emb = model(**inputs)
        scores = torch.nn.functional.cosine_similarity(candidates_emb, pos_docs_emb)

        if args.scaling_type=="minmax":
            min_temp=torch.min(scores)
            max_temp=torch.max(scores)
            if min>min_temp:
                min=min_temp
            if max<max_temp:
                max=max_temp
        elif args.scaling_type=="standard":
            for p in range(len(scores)):
                score_list.append(scores[p].cpu().item())


        if args.get_probs=='true':
            score_cand = [(scores[i].cpu().item(), candidates[i], probabilities[i]) for i in range(len(candidates))]
        else:
            score_cand = [(scores[i].cpu().item(), candidates[i]) for i in range(len(candidates))]
        score_cand_sorted = sorted(score_cand, key=lambda x: x[0], reverse=True)
        # candidates_sorted = [x[1] for x in score_cand_sorted]
        # item['model_generated_query_candidates_and_scores'] = score_cand_sorted
        item['model_generated_query_candidates'] = score_cand_sorted
        item['best_candidate']=score_cand_sorted[0][1]
        out.append(item)

    if args.scaling_type=="minmax":
        print("Min is: ",min,"  Max is: ", max)
        min=min.item()
        max=max.item()
    elif args.scaling_type=="standard":
        mean=torch.mean(torch.tensor(score_list)).item()
        std=torch.std(torch.tensor(score_list)).item()
        print("Mean is: ",mean,"  Std is: ", std)


    for m in tqdm(range(len(data))):
        record=out[m]
        coss_sim_score = [record["model_generated_query_candidates"][i][0] for i in range(len(record["model_generated_query_candidates"]))]
        candidates = [record["model_generated_query_candidates"][i][1] for i in range(len(record["model_generated_query_candidates"]))]
        if args.get_probs=='true':
            probabilities = [record["model_generated_query_candidates"][i][2] for i in range(len(record["model_generated_query_candidates"]))]
        if args.scaling_type=="minmax":
            coss_sim_score = [(coss_sim_score[i]-min)/(max-min) for i in range(len(coss_sim_score))]
        elif args.scaling_type=="standard":
            coss_sim_score = [(coss_sim_score[i]-mean)/std for i in range(len(coss_sim_score))]
        if args.get_probs=='true':
            score_cand = [(coss_sim_score[i], candidates[i], probabilities[i]) for i in range(len(candidates))]
        else:
            score_cand = [(coss_sim_score[i], candidates[i]) for i in range(len(candidates))]
        record['model_generated_query_candidates']=score_cand
        fw.write(json.dumps(record)+'\n')






    # if args.scaling_type=="minmax":
    #     for i in tqdm(range(len(data))):
    #         item=orig_data[i]
    #         cand=data[i]
    #         example_id = item['id']
    #         candidates = cand['oracle_utt_text']
    #         probabilities = cand['probability']
    #         #candidates = item['model_generated_query_candidates']
    #         pos_docs_emb = embs[example_id].cuda()

    #         inputs = tokenizer(candidates, padding=True, truncation=True, return_tensors="pt")
    #         inputs = inputs.to('cuda')
    #         candidates_emb = model(**inputs)

    #         scores = torch.nn.functional.cosine_similarity(candidates_emb, pos_docs_emb)
    #         min_temp=torch.min(scores)
    #         max_temp=torch.max(scores)
    #         if min>min_temp:
    #             min=min_temp
    #         if max<max_temp:
    #             max=max_temp
    #         score_cand = [(scores[i].cpu().item(), candidates[i], probabilities[i]) for i in range(len(candidates))]
    #         score_cand_sorted = sorted(score_cand, key=lambda x: x[0], reverse=True)
    #         # candidates_sorted = [x[1] for x in score_cand_sorted]
    #         # item['model_generated_query_candidates_and_scores'] = score_cand_sorted
    #         item['model_generated_query_candidates'] = score_cand_sorted
    #         item['best_candidate']=score_cand_sorted[0][1]
    #         out.append(item)
    #     print("Min is: ",min,"  Max is: ", max)
    #     min=min.item()
    #     max=max.item()
    #     for m in tqdm(range(len(data))):
    #         record=out[m]
    #         coss_sim_score = [record["model_generated_query_candidates"][i][0] for i in range(len(record["model_generated_query_candidates"]))]
    #         candidates = [record["model_generated_query_candidates"][i][1] for i in range(len(record["model_generated_query_candidates"]))]
    #         probabilities = [record["model_generated_query_candidates"][i][2] for i in range(len(record["model_generated_query_candidates"]))]
    #         coss_sim_score = [(coss_sim_score[i]-min)/(max-min) for i in range(len(coss_sim_score))]
    #         score_cand = [(coss_sim_score[i], candidates[i], probabilities[i]) for i in range(len(candidates))]
    #         record['model_generated_query_candidates']=score_cand
    #         fw.write(json.dumps(record)+'\n')
    # elif args.scaling_type=="standard":
    #     for i in tqdm(range(len(data))):
    #         item=orig_data[i]
    #         cand=data[i]
    #         example_id = item['id']
    #         candidates = cand['oracle_utt_text']
    #         probabilities = cand['probability']
    #         #candidates = item['model_generated_query_candidates']
    #         pos_docs_emb = embs[example_id].cuda()

    #         inputs = tokenizer(candidates, padding=True, truncation=True, return_tensors="pt")
    #         inputs = inputs.to('cuda')
    #         candidates_emb = model(**inputs)

    #         scores = torch.nn.functional.cosine_similarity(candidates_emb, pos_docs_emb)
    #         for p in range(len(scores)):
    #             score_list.append(scores[p].cpu().item())
    #         score_cand = [(scores[i].cpu().item(), candidates[i], probabilities[i]) for i in range(len(candidates))]
    #         score_cand_sorted = sorted(score_cand, key=lambda x: x[0], reverse=True)
    #         # candidates_sorted = [x[1] for x in score_cand_sorted]
    #         # item['model_generated_query_candidates_and_scores'] = score_cand_sorted
    #         item['model_generated_query_candidates'] = score_cand_sorted
    #         item['best_candidate']=score_cand_sorted[0][1]
    #         out.append(item)
    #     mean=torch.mean(torch.tensor(score_list)).item()
    #     std=torch.std(torch.tensor(score_list)).item()
    #     print("Mean is: ",mean,"  Std is: ", std)
    #     for m in tqdm(range(len(data))):
    #         record=out[m]
    #         coss_sim_score = [record["model_generated_query_candidates"][i][0] for i in range(len(record["model_generated_query_candidates"]))]
    #         candidates = [record["model_generated_query_candidates"][i][1] for i in range(len(record["model_generated_query_candidates"]))]
    #         probabilities = [record["model_generated_query_candidates"][i][2] for i in range(len(record["model_generated_query_candidates"]))]
    #         coss_sim_score = [(coss_sim_score[i]-mean)/std for i in range(len(coss_sim_score))]
    #         score_cand = [(coss_sim_score[i], candidates[i], probabilities[i]) for i in range(len(candidates))]
    #         record['model_generated_query_candidates']=score_cand
    #         fw.write(json.dumps(record)+'\n')