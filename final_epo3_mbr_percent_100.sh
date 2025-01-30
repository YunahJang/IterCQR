#!/bin/bash
# candidate generation 
export PYTHONPATH="../":"${PYTHONPATH}"
export JAVA_HOME=~/LG/jdk-17.0.7+7
export PATH=$PATH:$JAVA_HOME/bin

for i in {1..15}
do
  iter=$i
  prev_iter=$((i-1))
  log_dir_path="/itercqr/logs"
  topiocqa_test=f"/itercqr/data/datasets/topiocqa/dev_new.json"
  topiocqa_train="/itercqr/data/datasets/topiocqa/train_new.json"
  qrecc_test="/itercqr/data/new_preprocessed/test.json"
  decode_type="oracle"
  test_dataset="topiocqa"
  qrecc_test_dataset="qrecc"
  log_dir_path="/itercqr/logs"
  model_type="T5-base"
  selection_type="cossim"
  num_epoch=5
  ckpt_output_path="/itercqr/models/iterative/topiocqa/scst_comparison/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32"
  ckpt_trained_path="/itercqr/models/iterative/topiocqa/scst_comparison/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32/KD-ANCE-prefix-oracle/${num_epoch}"
  model_output_path="/itercqr/output/topiocqa/iterative/scst_comparison/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.json"
  initial_model_checkpoint_path="/itercqr/models/iterative/topiocqa/scst_comparison/topiocqa_trained_mbr_iter${prev_iter}_ver_epo3_ql32/KD-ANCE-prefix-oracle/${num_epoch}"

  candidate_file_path="/itercqr/data/datasets/topiocqa/iterative/topiocqa_trained_mbr_ver_epo3_ql32_from_iter${prev_iter}_for_iter_${iter}.json"
  best_candiate_path="/itercqr/data/datasets/topiocqa/iterative/best_candidates/scst_comparison/topiocqa_trained_mbr_ver_epo3_ql32_from_iter${prev_iter}_for_iter_${iter}.json"
    
  trec_file_name="topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.trec"
  qrel_output_path="/itercqr/output/topiocqa/iterative/final/${selection_type}/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32"
  qrecc_model_output_path="/itercqr/output/qrecc/zeroshot_iterative/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.json"
  qrecc_trec_file_name="topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.trec"
  qrecc_qrel_output_path="/itercqr/output/qrecc/zeroshot_iterative/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32"
  qrecc_qrel_output_name="qrecc_iterative_2step_beam_top1_mbr_iter${iter}_from_ckpt_cand10_nomse_mse2_novalid_epo${num_epoch}_nomse.json"



  result_save_path="/itercqr/test_results/comparison_scst/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.txt"

  scaling_type="minmax"
  kd_loss="false"
  if [ $iter == 0 ]
  then
    echo "***************THIS IS ITER $iter*****************"
    num_epoch=5
    #initial_model_checkpoint_path="/itercqr/models/topiocqa_gpt_rewrite_query_generation/KD-ANCE-prefix-oracle-best-model"
    ckpt_output_path="/itercqr/models/iterative/topiocqa/topiocqa_iter${iter}_epo${num_epoch}_nomse"
    ckpt_trained_path="/itercqr/models/iterative/topiocqa/topiocqa_iter0_epo15_nomse/KD-ANCE-prefix-oracle/3"
    topiocqa_gpt="/itercqr/data/datasets/topiocqa/topiocqa_rewrite_train_curr.json"
    echo "TRAIN WITH BEST CANDIDATE"
    python /itercqr/src/train_GQR_novalidation_grad_accum.py \
      --pretrained_query_encoder_tokenizer=$model_type \
      --pretrained_query_encoder="T5-base" \
      --pretrained_passage_encoder="castorini/ance-msmarco-passage" \
      --train_file_path=$topiocqa_train \
      --log_dir_path=$log_dir_path \
      --model_output_path=$ckpt_output_path \
      --collate_fn_type="flat_concat_for_train" \
      --decode_type=$decode_type \
      --per_gpu_train_batch_size=8 \
      --num_train_epochs=15 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --max_concat_length=512 \
      --train_dataset="topiocqa_gpt" \
      --num_candidates_for_training 10 \
      --alpha=0.5 \
      --use_data_percent=1 \
      --gradient_accumulation_steps 1 \
      --kd_loss=$kd_loss


    echo "GENERATE OUTPUT OF TRAINED MODEL"
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$ckpt_trained_path \
      --test_file_path=$topiocqa_test \
      --output_file_path=$model_output_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=32 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 1 \
      --num_return_sequences 1 \
      --top_k 50 \
      --top_p 0.9


    echo "GENERATE OUTPUT OF TRAINED MODEL FOR QRECC"
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$ckpt_trained_path \
      --test_file_path=$qrecc_test \
      --output_file_path=$qrecc_model_output_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=32 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $qrecc_test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 1 \
      --num_return_sequences 1 \
      --top_k 50 \
      --top_p 0.9


    cd ..

    cd src
    echo "EVALUATE TOPIOCQA DENSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python test_topiocqa_args.py \
      --test_file_path=$model_output_path \
      --qrel_output_path=$qrel_output_path \
      --result_save_path=$result_save_path

    echo "EVALUATE QRECC (ZERO-SHOT) DENSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python test_qrecc_args.py \
      --test_file_path=$qrecc_model_output_path \
      --qrel_output_path=$qrecc_qrel_output_path \
      --qrel_output_name=$qrecc_qrel_output_name \
      --result_save_path=$result_save_path
    cd ..

    cd bm25
    echo "EVALUATE TOPIOCQA SPARSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python /itercqr/bm25/bm25_topiocqa.py \
      --input_query_path=$model_output_path \
      --trec_file_name=$trec_file_name \
      --result_save_path=$result_save_path

    echo "EVALUATE QRECC (ZERO-SHOT) SPARSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python /itercqr/bm25/bm25_qrecc.py \
      --input_query_path=$qrecc_model_output_path \
      --trec_file_name=$qrecc_trec_file_name \
      --result_save_path=$result_save_path
    cd ..

    cd bashfile

  
  elif [ $iter == 1 ]
  then
    echo "***************THIS IS ITER $iter*****************"
    num_epoch=2
    ckpt_trained_path="/itercqr/models/iterative/topiocqa/scst_comparison/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32/KD-ANCE-prefix-oracle/${num_epoch}"
    model_output_path="/itercqr/output/topiocqa/iterative/scst_comparison/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.json"

    candidate_file_path="/itercqr/data/datasets/topiocqa/iterative/topiocqa_trained_mbr_ver_epo3_ql32_from_iter${prev_iter}_for_iter_${iter}.json"
    best_candiate_path="/itercqr/data/datasets/topiocqa/iterative/best_candidates/scst_comparison/topiocqa_trained_mbr_ver_epo3_ql32_from_iter${prev_iter}_for_iter_${iter}.json"
    
    trec_file_name="topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.trec"
    qrel_output_path="/itercqr/output/topiocqa/iterative/final/${selection_type}/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32"
    qrecc_model_output_path="/itercqr/output/qrecc/zeroshot_iterative/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.json"
    qrecc_trec_file_name="topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32.trec"
    qrecc_qrel_output_path="/itercqr/output/qrecc/zeroshot_iterative/topiocqa_trained_mbr_iter${iter}_ver_epo3_ql32"
    qrecc_qrel_output_name="qrecc_iterative_2step_beam_top1_mbr_iter${iter}_from_ckpt_cand10_nomse_mse2_novalid_epo${num_epoch}_nomse.json"


    initial_model_checkpoint_path="/itercqr/models/iterative/topiocqa/topiocqa_iter0_epo15_nomse/KD-ANCE-prefix-oracle/3"
    echo "CANDIDATE GENERATION"
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$initial_model_checkpoint_path \
      --test_file_path=$topiocqa_train \
      --output_file_path=$candidate_file_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=8 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 10 \
      --num_return_sequences 10 \
      --top_k 50 \
      --top_p 0.9


    candidate selection

    if [ "$selection_type" == "cossim" ]
    then
        echo "candidate_selection by cossim"
        python /itercqr/data/datasets/topiocqa/get_candidate_cos_sim_score_scale_final.py \
            --output_path=$best_candiate_path \
            --candidate_path=$candidate_file_path \
            --scaling_type=$scaling_type
    else 
        echo "candidate_selection by HISTORY"
        python /itercqr/src/candidate_selection.py \
            --output_path=$best_candiate_path \
            --candidate_path=$candidate_file_path
    fi
    echo "TRAIN WITH BEST CANDIDATE"
    python /itercqr/src/train_GQR_novalidation_grad_accum.py \
      --pretrained_query_encoder_tokenizer=$model_type \
      --pretrained_query_encoder=$initial_model_checkpoint_path \
      --pretrained_passage_encoder="castorini/ance-msmarco-passage" \
      --train_file_path=$best_candiate_path \
      --log_dir_path=$log_dir_path \
      --model_output_path=$ckpt_output_path \
      --collate_fn_type="flat_concat_for_train" \
      --decode_type=$decode_type \
      --per_gpu_train_batch_size=4 \
      --num_train_epochs=2 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --max_concat_length=512 \
      --train_dataset="topiocqa_mbr" \
      --num_candidates_for_training 10 \
      --alpha=0.5 \
      --use_data_percent=1 \
      --gradient_accumulation_steps 2 \
      --kd_loss=$kd_loss


    echo "GENERATE OUTPUT OF TRAINED MODEL"
    #generate trained model's output 
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$ckpt_trained_path \
      --test_file_path=$topiocqa_test \
      --output_file_path=$model_output_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=32 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 1 \
      --num_return_sequences 1 \
      --top_k 50 \
      --top_p 0.9


    echo "GENERATE OUTPUT OF TRAINED MODEL FOR QRECC"
    #generate trained model's output 
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$ckpt_trained_path \
      --test_file_path=$qrecc_test \
      --output_file_path=$qrecc_model_output_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=32 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $qrecc_test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 1 \
      --num_return_sequences 1 \
      --top_k 50 \
      --top_p 0.9


    cd ..

    cd src
    echo "EVALUATE TOPIOCQA DENSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python test_topiocqa_args.py \
      --test_file_path=$model_output_path \
      --qrel_output_path=$qrel_output_path \
      --result_save_path=$result_save_path

    echo "EVALUATE QRECC (ZERO-SHOT) DENSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python test_qrecc_args.py \
      --test_file_path=$qrecc_model_output_path \
      --qrel_output_path=$qrecc_qrel_output_path \
      --qrel_output_name=$qrecc_qrel_output_name \
      --result_save_path=$result_save_path
    cd ..

    cd bm25
    echo "EVALUATE TOPIOCQA SPARSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python /itercqr/bm25/bm25_topiocqa.py \
      --input_query_path=$model_output_path \
      --trec_file_name=$trec_file_name \
      --result_save_path=$result_save_path

    echo "EVALUATE QRECC (ZERO-SHOT) SPARSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python /itercqr/bm25/bm25_qrecc.py \
      --input_query_path=$qrecc_model_output_path \
      --trec_file_name=$qrecc_trec_file_name \
      --result_save_path=$result_save_path
    cd ..

    cd bashfile

  elif [ $iter == 2 ]
  then
  
    echo "***************THIS IS ITER $iter*****************"
    num_epoch=5
    initial_model_checkpoint_path="/itercqr/models/iterative/topiocqa/scst_comparison/topiocqa_trained_mbr_iter${prev_iter}_ver_epo3_ql32/KD-ANCE-prefix-oracle/2"

    echo "CANDIDATE GENERATION"
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$initial_model_checkpoint_path \
      --test_file_path=$topiocqa_train \
      --output_file_path=$candidate_file_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=8 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 10 \
      --num_return_sequences 10 \
      --top_k 50 \
      --top_p 0.9


    echo candidate selection

    if [ "$selection_type" == "cossim" ]
    then
        echo "candidate_selection by cossim"
        python /itercqr/data/datasets/topiocqa/get_candidate_cos_sim_score_scale_final.py \
            --output_path=$best_candiate_path \
            --candidate_path=$candidate_file_path \
            --scaling_type=$scaling_type
    else 
        echo "candidate_selection by HISTORY"
        python /itercqr/src/candidate_selection.py \
            --output_path=$best_candiate_path \
            --candidate_path=$candidate_file_path
    fi

    train with best candidate 
    echo "TRAIN WITH BEST CANDIDATE"
    python /itercqr/src/train_GQR_novalidation_grad_accum.py \
      --pretrained_query_encoder_tokenizer=$model_type \
      --pretrained_query_encoder=$initial_model_checkpoint_path \
      --pretrained_passage_encoder="castorini/ance-msmarco-passage" \
      --train_file_path=$best_candiate_path \
      --log_dir_path=$log_dir_path \
      --model_output_path=$ckpt_output_path \
      --collate_fn_type="flat_concat_for_train" \
      --decode_type=$decode_type \
      --per_gpu_train_batch_size=8 \
      --num_train_epochs=$num_epoch \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --max_concat_length=512 \
      --train_dataset="topiocqa_iterative" \
      --num_candidates_for_training 1 \
      --alpha=0.5 \
      --use_data_percent=1 \
      --gradient_accumulation_steps 1 \
      --kd_loss=$kd_loss


    echo "GENERATE OUTPUT OF TRAINED MODEL"
    #generate trained model's output 
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$ckpt_trained_path \
      --test_file_path=$topiocqa_test \
      --output_file_path=$model_output_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=32 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 1 \
      --num_return_sequences 1 \
      --top_k 50 \
      --top_p 0.9


    echo "GENERATE OUTPUT OF TRAINED MODEL FOR QRECC"
    #generate trained model's output 
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$ckpt_trained_path \
      --test_file_path=$qrecc_test \
      --output_file_path=$qrecc_model_output_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=32 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $qrecc_test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 1 \
      --num_return_sequences 1 \
      --top_k 50 \
      --top_p 0.9


    cd ..

    cd src
    echo "EVALUATE TOPIOCQA DENSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python test_topiocqa_args.py \
      --test_file_path=$model_output_path \
      --qrel_output_path=$qrel_output_path \
      --result_save_path=$result_save_path

    echo "EVALUATE QRECC (ZERO-SHOT) DENSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python test_qrecc_args.py \
      --test_file_path=$qrecc_model_output_path \
      --qrel_output_path=$qrecc_qrel_output_path \
      --qrel_output_name=$qrecc_qrel_output_name \
      --result_save_path=$result_save_path
    cd ..

    cd bm25
    echo "EVALUATE TOPIOCQA SPARSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python /itercqr/bm25/bm25_topiocqa.py \
      --input_query_path=$model_output_path \
      --trec_file_name=$trec_file_name \
      --result_save_path=$result_save_path

    echo "EVALUATE QRECC (ZERO-SHOT) SPARSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python /itercqr/bm25/bm25_qrecc.py \
      --input_query_path=$qrecc_model_output_path \
      --trec_file_name=$qrecc_trec_file_name \
      --result_save_path=$result_save_path
    cd ..

    cd bashfile

  else
    echo "***************THIS IS ITER $iter*****************"
    num_epoch=5
    echo "CANDIDATE GENERATION"
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$initial_model_checkpoint_path \
      --test_file_path=$topiocqa_train \
      --output_file_path=$candidate_file_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=8 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 10 \
      --num_return_sequences 10 \
      --top_k 50 \
      --top_p 0.9

    echo "CANDIDATE SELECTION"

    if [ "$selection_type" == "cossim" ]
    then
        echo "candidate_selection by cossim"
        python /itercqr/data/datasets/topiocqa/get_candidate_cos_sim_score_scale_final.py \
            --output_path=$best_candiate_path \
            --candidate_path=$candidate_file_path \
            --scaling_type=$scaling_type
    else 
        echo "candidate_selection by HISTORY"
        python /itercqr/src/candidate_selection.py \
            --output_path=$best_candiate_path \
            --candidate_path=$candidate_file_path
    fi

    echo "TRAIN WITH BEST CANDIDATE"
    python /itercqr/src/train_GQR_novalidation_grad_accum.py \
      --pretrained_query_encoder_tokenizer=$model_type \
      --pretrained_query_encoder=$initial_model_checkpoint_path \
      --pretrained_passage_encoder="castorini/ance-msmarco-passage" \
      --train_file_path=$best_candiate_path \
      --log_dir_path=$log_dir_path \
      --model_output_path=$ckpt_output_path \
      --collate_fn_type="flat_concat_for_train" \
      --decode_type=$decode_type \
      --per_gpu_train_batch_size=8 \
      --num_train_epochs=$num_epoch \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --max_concat_length=512 \
      --train_dataset="topiocqa_iterative" \
      --num_candidates_for_training 1 \
      --alpha=0.5 \
      --use_data_percent=1 \
      --gradient_accumulation_steps 1 \
      --kd_loss=$kd_loss


    echo "GENERATE OUTPUT OF TRAINED MODEL"
    #generate trained model's output 
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$ckpt_trained_path \
      --test_file_path=$topiocqa_test \
      --output_file_path=$model_output_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=32 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 1 \
      --num_return_sequences 1 \
      --top_k 50 \
      --top_p 0.9


    echo "GENERATE OUTPUT OF TRAINED MODEL FOR QRECC"
    #generate trained model's output 
    python /itercqr/src/test_GQR.py \
      --model_checkpoint_path=$ckpt_trained_path \
      --test_file_path=$qrecc_test \
      --output_file_path=$qrecc_model_output_path \
      --collate_fn_type="flat_concat_for_test" \
      --decode_type=$decode_type \
      --per_gpu_eval_batch_size=32 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --test_dataset $qrecc_test_dataset \
      --max_concat_length=512 \
      --do_sample="false" \
      --num_beams 1 \
      --num_return_sequences 1 \
      --top_k 50 \
      --top_p 0.9


    cd ..

    cd src
    echo "EVALUATE TOPIOCQA DENSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python test_topiocqa_args.py \
      --test_file_path=$model_output_path \
      --qrel_output_path=$qrel_output_path \
      --result_save_path=$result_save_path

    echo "EVALUATE QRECC (ZERO-SHOT) DENSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python test_qrecc_args.py \
      --test_file_path=$qrecc_model_output_path \
      --qrel_output_path=$qrecc_qrel_output_path \
      --qrel_output_name=$qrecc_qrel_output_name \
      --result_save_path=$result_save_path
    cd ..

    cd bm25
    echo "EVALUATE TOPIOCQA SPARSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python /itercqr/bm25/bm25_topiocqa.py \
      --input_query_path=$model_output_path \
      --trec_file_name=$trec_file_name \
      --result_save_path=$result_save_path

    echo "EVALUATE QRECC (ZERO-SHOT) SPARSE RETRIEVAL RESULT FOR TRAINED MODEL"
    python /itercqr/bm25/bm25_qrecc.py \
      --input_query_path=$qrecc_model_output_path \
      --trec_file_name=$qrecc_trec_file_name \
      --result_save_path=$result_save_path
    cd ..

    cd bashfile
  fi
  
done