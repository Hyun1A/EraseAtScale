export CUDA_DEVICE_ORDER=PCI_BUS_ID


CONFIG="./configs/train_hetero/config_distill.yaml"
ST_IDX="0"
END_IDX="1000"
NOISE="0.001"
GATE_RANK_LIST=( "16" )
RAND_LIST=( "0.5" ) 

DEPTH_LIST=( "1" ) 
ACT_LIST=( "none" ) 
BIAS_LIST=( "True" ) 

PAL_LIST=( "300.0" )
ARCH_TYPE="moe_dense" # "mlp", "mlp_swiglu", "linear", "ffn"
GUIDE="0.5"
RANK_LIST=( "384" ) # 768, 1536 3072 4608 
NET_TYPE="ca_sel_block" # "ca_kv", "ca_v", "ca_sel_kv"
SAMP_TYPE=""

for ACT in "${ACT_LIST[@]}"; do
    for BIAS in "${BIAS_LIST[@]}"; do
        for RANK in "${RANK_LIST[@]}"; do
            for DEPTH in "${DEPTH_LIST[@]}"; do
                for GATE_RANK in "${GATE_RANK_LIST[@]}"; do
                    for PAL in "${PAL_LIST[@]}"; do
                        for RAND in "${RAND_LIST[@]}"; do
                            CUDA_VISIBLE_DEVICES="0" python ./train/train_distill_gate.py \
                                --config_file ${CONFIG} \
                                --st_prompt_idx ${ST_IDX} \
                                --end_prompt_idx ${END_IDX} \
                                --gate_rank ${GATE_RANK} \
                                --guidance_scale ${GUIDE} \
                                --pal ${PAL} \
                                --noise ${NOISE} \
                                --rand ${RAND} \
                                --skip_learned True \
                                --arch_type ${ARCH_TYPE} \
                                --depth ${DEPTH} \
                                --lora_rank ${RANK} \
                                --use_bias ${BIAS} \
                                --activation ${ACT} \
                                --num_shards 16 \
                                --iterations 8096 \
                                --dataset_n_batch 1 \
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
                                --model_domains "./output/Singleton_Celeb/celeb_1000" "./output/Singleton_Char/char_500" "./output/Singleton_Artist/artist_700" \
                                --model_path gate_guide0.0_pal30.0_gr16_c0.9_r4_map_dissimilar_3 \
                                --mapping_type "dissimilar" \
                                --n_top 3                                
                        done
                    done
                done
            done
        done
    done
done