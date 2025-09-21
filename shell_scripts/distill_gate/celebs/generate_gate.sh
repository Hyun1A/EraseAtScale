export CUDA_DEVICE_ORDER=PCI_BUS_ID

GEN_CONFIG=configs/gen_celeb/config_celeb_erased_small.yaml
GEN_ST_IDX=0
GEN_END_IDX=100000000
GUIDE="0.0"
NOISE="0.001"

GATE_RANK="16"
PAL_LIST=( "30.0" )
CONF="0.9"
RANK="4"
MAP_TYPE="similar_3"

for PAL in "${PAL_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="0" python ./generate/generate_gate.py --config ${GEN_CONFIG} \
        --model_domains "./output/Singleton_Celeb/celeb_1000" \
        --model_path gate_guide${GUIDE}_pal${PAL}_gr${GATE_RANK}_c${CONF}_r${RANK}_map_${MAP_TYPE} \
        --save_env gate_guide${GUIDE}_pal${PAL}_gr${GATE_RANK}_c${CONF}_r${RANK}_map_${MAP_TYPE} \
        --st_prompt_idx ${GEN_ST_IDX} \
        --end_prompt_idx ${GEN_END_IDX} \
        --gate_rank ${GATE_RANK}
done
