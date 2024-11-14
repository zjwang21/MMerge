#!/bin/bash
echo "----------------"
sleep 5
work_wait() {
    TRAIN_FLAG="megatron"
    sleep 1500
    while true; do
        if  ps -ef | grep "$TRAIN_FLAG" | grep -v "grep" > /dev/null
        then
            echo "training..."
            sleep 30
        else
            exit 1
        fi
    done
}

echo $RANK
echo $MASTER_ADDR
sleep 5

export PATH=$PATH:/root/.local/bin
echo $PATH
ENV_DIR=/root/work/huangxin/envs/nju-megatron
WORK_DIR=/root/work/huangxin/nanda/ImplicitTransBridge-master

if [ ${RANK} -eq 0 ];
then
export PATH=$PATH:$ENV_DIR/bin
python /root/work/huangxin/nanda/SuperAlign/ds_configs/hostfile.py
export DLTS_HOSTFILE=/root/work/huangxin/nanda/SuperAlign/ds_configs/multi-node-hostfile
bash $WORK_DIR/scripts/stage0.sh
work_wait
fi