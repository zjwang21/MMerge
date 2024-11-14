PROJECT_PATH=/root/work/huangxin/nanda/ImplicitTransBridge-master/eval
MODEL_PATH=/root/work/huangxin/nanda/models/Qwen/Qwen2.5-Math-7B-Instruct

# For 13B model, you may need to set batch_size smaller, like 16, to avoid OOM issue.
python $PROJECT_PATH/mgsm_test.py \
    --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 32 \
    --lang_only English