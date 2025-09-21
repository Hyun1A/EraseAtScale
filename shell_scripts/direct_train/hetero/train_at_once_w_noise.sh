export CUDA_DEVICE_ORDER=PCI_BUS_ID

CONFIG="./configs/train_hetero/config_moe.yaml"
ST_IDX="0"
END_IDX="1000"
NOISE="0.001"
GATE_RANK="16"

DEPTH_LIST=( "1" )
ACT="none"
BIAS="True"

PAL_LIST=( "300.0" )
ARCH_TYPE="moe_dense" # "mlp", "mlp_swiglu", "linear", "ffn"
GUIDE="0.5"
RANK_LIST=( "384" ) # 768, 1536 3072 4608
NET_TYPE="ca_sel_block" # "ca_kv", "ca_v", "ca_sel_kv"

RAND="0.5"
NOISE_TYPE="low_rank_1"

for RANK in "${RANK_LIST[@]}"; do
    for DEPTH in "${DEPTH_LIST[@]}"; do
        for PAL in "${PAL_LIST[@]}"; do
            CUDA_VISIBLE_DEVICES="0" python ./train/train_at_once.py \
                --config_file ${CONFIG} \
                --st_prompt_idx ${ST_IDX} \
                --end_prompt_idx ${END_IDX} \
                --gate_rank ${GATE_RANK} \
                --guidance_scale ${GUIDE} \
                --pal ${PAL} \
                --noise ${NOISE} \
                --skip_learned True \
                --arch_type ${ARCH_TYPE} \
                --depth ${DEPTH} \
                --lora_rank ${RANK} \
                --use_bias ${BIAS} \
                --activation ${ACT} \
                --num_shards 16 \
                --iterations 16192 \
                --dataset_n_batch 512 \
                --lr 0.01 \
                --dataset_n_anc 4 \
                --dataset_seed 42 \
                --conf 0.9 \
                --net_type ${NET_TYPE} \
                --glu_type glu \
                --n_experts 8 \
                --top_k 1 \
                --keeptok 0.1 \
                --router_noise_std 0.1 \
                --sparse_compute True \
                --lb_coef 1e-2 \
                --cov_coef 1e-3 \
                --ffn_norm rmsnorm \
                --moe_dropout 0.0 \
                --moe_resid_dropout 0.0 \
                --prenorm_router true \
                --moe_aux_coef 1.0 \
                --model_domains "./output/Singleton_Celeb/celeb_1000" "./output/Singleton_Char/char_400" "./output/Singleton_Artist/artist_700" \
                --model_path gate_guide1.0_pal30.0_gate_rank16_last_token_conf0.9 \
                --mapping_type "similar" \
                --n_top 3 \
                --rand ${RAND} \
                --noise_type ${NOISE_TYPE}
        done
    done
done