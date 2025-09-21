export CUDA_DEVICE_ORDER=PCI_BUS_ID

GEN_CONFIG=configs/gen_explicit/config_explicit_small.yaml
GEN_ST_IDX=0
GEN_END_IDX=100000000
GUIDE="0.0"
NOISE="0.001"

GATE_RANK="16"
PAL_LIST=( "30.0" )

for PAL in "${PAL_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="0" python ./generate/generate_eas_gate.py --config ${GEN_CONFIG} \
        --model_path ./output/Singleton_Explicit/explicit/gate_guide${GUIDE}_pal${PAL}_gate_rank${GATE_RANK}_conf0.95_batch256_iter5400_a_person_in_the_nude_to_modest_clothes \
        --save_env gate_guide${GUIDE}_pal${PAL}_gate_rank${GATE_RANK}_erase_${CONCEPT}_conf0.95_batch256_iter5400_a_person_in_the_nude_to_modest_clothes \
        --st_prompt_idx ${GEN_ST_IDX} \
        --end_prompt_idx ${GEN_END_IDX} \
        --gate_rank ${GATE_RANK}
done
