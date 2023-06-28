# Set visible devices
export CUDA_VISIBLE_DEVICES="0"

# Set MP_SIZE to the number of devices
MP_SIZE=1

# Provide path to vocab file and model
#ROOT="/YaLM-100B-4bit"
#ROOT="/YaLM-100B/download/yalm100b_checkpoint"
ROOT="data"
VOCAB_PATH="${ROOT}/vocab/voc_100b.sp"
MODEL_PATH="${ROOT}/weights"
LOAD_ARGS="\
    --vocab-file ${VOCAB_PATH} \
    --load ${MODEL_PATH}"

# Set generation parameters
GEN_ARGS="
    --temperature 1.0 \
    --top_p 0.9 \
    --seed 1234 \
    --seq-length 256 \
    --out-seq-length 128"

HPARAM_ARGS="\
    --pos-encoding-type rotary \
    --num-layers 80 \
    --embedding-size 512 \
    --hidden-size 10240 \
    --intermediate-size 27308 \
    --activation-type geglu \
    --num-attention-heads 128 \
    --max-position-embeddings 1024 \
    --tokenizer-type SentencePiece \
    --fp16"

#DISTRIBUTED_ARGS="\
#    --nproc_per_node $MP_SIZE \
#    --nnodes 1 \
#    --node_rank 0 \
#    --master_addr localhost \
#    --master_port=1234"

COMMON_ARGS="\
    --num-samples 0 \
    --load-release-checkpoint \
    --batch-size 1 \
    --model-parallel-size $MP_SIZE \
    --make-vocab-size-divisible-by 1"

python3 predict.py $LOAD_ARGS $HPARAM_ARGS $GEN_ARGS #$COMMON_ARGS

#torchrun $DISTRIBUTED_ARGS predict.py \
#    $LOAD_ARGS \
#    $HPARAM_ARGS \
#    $COMMON_ARGS \
#    $GEN_ARGS
