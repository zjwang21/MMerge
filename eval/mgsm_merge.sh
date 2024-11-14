PROJECT_PATH=/root/work/huangxin/nanda/ImplicitTransBridge-master/eval
MODEL_PATH=/root/work/huangxin/nanda/ImplicitTransBridge-master/checkpoints/mt5xl-aug-qwe2.5-math7b-metamath
export PYTHONPATH=/root/work/huangxin/nanda/ImplicitTransBridge-master:$PYTHONPATH
# For 13B model, you may need to set batch_size smaller, like 16, to avoid OOM issue.
python $PROJECT_PATH/eval_mgsm.py \
    --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 1 \
    --lang_only Bengali Thai Swahili Japanese Chinese German French Russian Spanish English