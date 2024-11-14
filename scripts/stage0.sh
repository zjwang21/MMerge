data_path=/root/work/huangxin/nanda/QAlign-master/data/metamath/MetaMathQA-395K.json
llm_path=/root/work/huangxin/nanda/models/Qwen/Qwen2.5-Math-7B-Instruct
slm_path_a=/root/work/huangxin/nanda/models/google/mt5-xl-lm-adapt
slm_path_b=/root/work/huangxin/nanda/models/google/mt5-xl-lm-adapt
output_path=/root/work/huangxin/nanda/ImplicitTransBridge-master/checkpoints/mt5xl-aug-qwe2.5-math7b-metamath

# 数据格式：json
# {"en": xxx, "zh": xxx, "bn": xxx, ,,,,,,}
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#deepspeed --num_gpus 8 --master_port=9901 /root/work/huangxin/nanda/ImplicitTransBridge-master/train.py \
#    --deepspeed /root/work/huangxin/nanda/ImplicitTransBridge-master/ds_configs/ds_config.json \
deepspeed --num_gpus 8 --master_port=9901 /root/work/huangxin/nanda/ImplicitTransBridge-master/train.py \
    --deepspeed /root/work/huangxin/nanda/ImplicitTransBridge-master/ds_configs/ds_config.json \
    --dataset_name $data_path \
    --preprocessing_num_workers 64 \
    --llm_path $llm_path \
    --slm_path_a $slm_path_a \
    --slm_path_b $slm_path_b \
    --stage 0 \
    --output_dir $output_path \
    --do_train \
    --max_seq_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 6e-4 \
    --num_train_epochs 1 \
    --save_only_model \
    --logging_steps 10 \
    --save_steps 2000 \
    --seed 42 \
    --overwrite_output_dir \
    --bf16 > /root/work/huangxin/nanda/ImplicitTransBridge-master/mt5-xl-aug-metamath-6e-4.out 2>&1