export CUDA_DEVICE_ORDER=PCI_BUS_ID

GEN_CONFIG=configs/gen_hetero/config_hetero_dev_extended.yaml
GEN_ST_IDX=0
GEN_END_IDX=100000000

CUDA_VISIBLE_DEVICES="0" python ./generate/generate_org.py --config ${GEN_CONFIG} \
    --st_prompt_idx ${GEN_ST_IDX} \
    --end_prompt_idx ${GEN_END_IDX} \
    --save_env org \
