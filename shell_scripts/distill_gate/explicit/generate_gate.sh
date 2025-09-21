export CUDA_DEVICE_ORDER=PCI_BUS_ID

GEN_CONFIG=configs/gen_explicit/config_explicit_small.yaml
GEN_ST_IDX=0
GEN_END_IDX=100000000
GUIDE="0.0"
NOISE="0.001"

GATE_RANK="64"
RANK="64"
CONF=0.95
PAL_LIST=( "30.0" )




for PAL in "${PAL_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="0" python ./generate/generate_eas_gate.py --config ${GEN_CONFIG} \
        --model_path ./output/Singleton_Explicit_1/gate_guide${GUIDE}_pal${PAL}_gate_rank${GATE_RANK}_rank${RANK}_last_token_conf${CONF} \
        --save_env gate_guide${GUIDE}_pal${PAL}_gate_rank${GATE_RANK}_rank${RANK}_last_token_conf${CONF} \
        --st_prompt_idx ${GEN_ST_IDX} \
        --end_prompt_idx ${GEN_END_IDX} \
        --gate_rank ${GATE_RANK}
done
