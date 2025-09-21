export CUDA_DEVICE_ORDER=PCI_BUS_ID


CONFIG="./configs/train_celeb/config_gate.yaml"
GUIDE="0.0"
ST_IDX="10"
END_IDX="1000"
NOISE="0.001"
ARCH_TYPE="gate"


GATE_RANK_LIST=( "16" )
PAL_LIST=( "30.0" )


for GATE_RANK in "${GATE_RANK_LIST[@]}"; do
    for PAL in "${PAL_LIST[@]}"; do
        CUDA_VISIBLE_DEVICES="0" python ./train/extract_last_token.py \
            --config_file ${CONFIG} \
            --st_prompt_idx ${ST_IDX} \
            --end_prompt_idx ${END_IDX} \
            --gate_rank ${GATE_RANK} \
            --guidance_scale ${GUIDE} \
            --pal ${PAL} \
            --rand 0.5 \
            --skip_learned True \
            --batch_size 8 \
            --arch_type ${ARCH_TYPE} \
            --conf 0.9 \
            --mapping_type "dissimilar" \
            --n_top 3
    done
done
