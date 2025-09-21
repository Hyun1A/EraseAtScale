export CUDA_DEVICE_ORDER=PCI_BUS_ID


# GEN_CONFIG=configs/gen_celeb/config_celeb_erased_small.yaml

GEN_CONFIG=configs/gen_hetero/config_hetero_dev_extended.yaml
GEN_ST_IDX=0
GEN_END_IDX=100000000
GUIDE="0.3"
NOISE="0.001"


GATE_RANK="16"
PAL_LIST=( "30.0" ) # "1000.0" "10000.0" "100000.0" )

RAND_LIST=( "0.5" ) 


for PAL in "${PAL_LIST[@]}"; do
    for RAND in "${RAND_LIST[@]}"; do
        CUDA_VISIBLE_DEVICES="0" python ./generate/generate_org.py --config ${GEN_CONFIG} \
            --save_env org \
            --st_prompt_idx ${GEN_ST_IDX} \
            --end_prompt_idx ${GEN_END_IDX} \
            --gate_rank ${GATE_RANK} \
            --rand ${RAND}
    done
done
