data_path=/root/work/huangxin/nanda/ImplicitTransBridge-master/data/opus_en_is.jsonl
llm_path=/root/work/huangxin/nanda/models/Qwen/Qwen2.5-7B
slm_path_a=/root/work/huangxin/nanda/LLaMA-Factory-main/saves/qwen2.5-0.5b/pretrain/is
slm_path_b=/root/work/huangxin/nanda/LLaMA-Factory-main/saves/qwen2.5-0.5b/pretrain/is
output_path=/root/work/huangxin/nanda/ImplicitTransBridge-master/checkpoints/stage1-test

# 数据格式：json
# {"src": xxx, "tgt": xxx}
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#deepspeed --num_gpus 2 --master_port=9901 train.py \
#    --deepspeed /root/work/huangxin/nanda/ImplicitTransBridge-master/ds_configs/ds_config.json \
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 /root/work/huangxin/nanda/ImplicitTransBridge-master/train.py \
    --dataset_name $data_path \
    --preprocessing_num_workers 16 \
    --llm_path $llm_path \
    --slm_path_a $slm_path_a \
    --slm_path_b $slm_path_b \
    --stage 1 \
    --src_lang English \
    --tgt_lang Icelandic \
    --output_dir $output_path \
    --do_train \
    --max_seq_length 200 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --save_only_model \
    --logging_steps 10 \
    --save_steps 2000 \
    --seed 42 \
    --overwrite_output_dir \
    --bf16 > /root/work/huangxin/nanda/ImplicitTransBridge-master/stage1.out 2>&1