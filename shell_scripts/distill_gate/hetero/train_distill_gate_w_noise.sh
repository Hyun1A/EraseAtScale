export CUDA_DEVICE_ORDER=PCI_BUS_ID


CONFIG="./configs/train_hetero/config_distill.yaml"
GUIDE="1.0"
ST_IDX="0"
END_IDX="1000"
NOISE="0.001"


GATE_RANK_LIST=( "16" )
RAND_LIST=( "0.5" ) 

ARCH_TYPE="ffn" # "mlp", "mlp_swiglu", "linear", "ffn"

PAL_LIST=( "100.0" )
DEPTH_LIST=( "1" ) 
RANK_LIST=( "1536" ) # 768, 1536 3072 4608 
ACT_LIST=( "none" ) 
BIAS_LIST=( "True" ) 
NOISE_TYPE="low_rank_16"



for ACT in "${ACT_LIST[@]}"; do
    for BIAS in "${BIAS_LIST[@]}"; do
        for RANK in "${RANK_LIST[@]}"; do
            for DEPTH in "${DEPTH_LIST[@]}"; do
                for GATE_RANK in "${GATE_RANK_LIST[@]}"; do
                    for PAL in "${PAL_LIST[@]}"; do
                        for RAND in "${RAND_LIST[@]}"; do
                            CUDA_VISIBLE_DEVICES="0" python ./train/train_distill_gate_last_token.py \
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
                                --noise_type ${NOISE_TYPE} \
                                --num_shards 16 \
                                --dataset_n_batch 128 \
                                --dataset_n_anc 4 \
                                --dataset_seed 42 \
                                --iterations 180000 \
                                --conf 0.9 \
                                --model_domains "./output/Singleton_Hetero/hetero_2000" \
                                --model_path gate_guide1.0_pal100.0_gate_rank16_depth1_last_token_conf0.9_rank1536
                        done
                    done
                done
            done
        done
    done
done