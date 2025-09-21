export CUDA_DEVICE_ORDER=PCI_BUS_ID


CONFIG="./configs/train_explicit/config_gate.yaml"
GUIDE="0.0"
ST_IDX="0"
END_IDX="0"
NOISE="0.001"
ARCH_TYPE="gate"

PAL_LIST=( "30.0" )
GATE_RANK_LIST=( "64" "128" )
RANK_LIST=( "64" "128" )
CONF_LIST=( "0.995" ) # "0.9" "0.95" "0.99" "0.995" )

for CONF in "${CONF_LIST[@]}"; do
    for RANK in "${RANK_LIST[@]}"; do
        for GATE_RANK in "${GATE_RANK_LIST[@]}"; do
            for PAL in "${PAL_LIST[@]}"; do
                CUDA_VISIBLE_DEVICES="0" python ./train/train_gate_last_token.py \
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
                    --conf ${CONF} \
                    --lora_rank ${RANK}
            done
        done
    done
done