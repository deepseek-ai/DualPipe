export OMP_NUM_THREADS=4
export MUSA_LAUNCH_BLOCKING=1
export MCCL_ALGOS=1
export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=3200000
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MCCL_IB_GID_INDEX=3
export MUSA_BLOCK_SCHEDULE_MODE=1
# export MCCL_BUFFSIZE=20480000
WORK_HOME="$PWD"
CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME

DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes 1
    --node_rank 0
    --master_addr 127.0.0.1
    --master_port 12345
    --log_dir $WORK_HOME/output/$CURRENT_TIME
    --redirects 3
)

cmd="PYTHONPATH=$PYTHONPATH:./dualpipe torchrun \
        ${DISTRIBUTED_ARGS[@]}
        example.py
    "
echo $cmd
eval $cmd
