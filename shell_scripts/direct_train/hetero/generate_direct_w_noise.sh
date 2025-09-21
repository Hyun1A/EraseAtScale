export CUDA_DEVICE_ORDER=PCI_BUS_ID

GEN_CONFIG=configs/gen_hetero/config_hetero_dev_extended.yaml
GEN_ST_IDX=0
GEN_END_IDX=100000000
NOISE="0.001"

GATE_RANK="16"
DEPTH="1" 
ACT="none" #"silu" "swish" "relu" "leaky_relu" "prelu" ) # "leaky_relu" "prelu" "elu" "selu" "gelu" "silu" "swish" "mish" )
BIAS="True" # "True" ) # "False")
CONF=0.9

ARCH_TYPE="moe_dense"
NET_TYPE="ca_sel_block"  # "ca_kv", "ca_v", "ca_sel_kv", "ca_sel_block"
GUIDE="0.5"
PAL_LIST=( "300.0" )
RANK="384"
ACT_TYPE="glu"
N_EXP=8
TOPK=1
KEEPTOP=0.1
BATCH="512"
LR="0.01"
ITER="16192"
MAP_TYPE="dissimilar_3"

for PAL in "${PAL_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES="0" python ./generate/generate_direct_hetero_w_noise.py --config ${GEN_CONFIG} \
        --model_domains "./output/Singleton_Hetero/hetero_2000" \
        --model_path direct_guide${GUIDE}_pal${PAL}_gr${GATE_RANK}_d${DEPTH}_c${CONF}_r${RANK}_map_${MAP_TYPE}_${NET_TYPE}_${ARCH_TYPE}_${ACT_TYPE}_E${N_EXP}_k${TOPK}_keep${KEEPTOP}_b${BATCH}_lr${LR}_it${ITER} \
        --save_env direct_guide${GUIDE}_pal${PAL}_gr${GATE_RANK}_d${DEPTH}_c${CONF}_r${RANK}_map_${MAP_TYPE}_${NET_TYPE}_${ARCH_TYPE}_${ACT_TYPE}_E${N_EXP}_k${TOPK}_keep${KEEPTOP}_b${BATCH}_lr${LR}_it${ITER} \
        --st_prompt_idx ${GEN_ST_IDX} \
        --end_prompt_idx ${GEN_END_IDX} \
        --gate_rank ${GATE_RANK} \
        --arch_type ${ARCH_TYPE} \
        --depth ${DEPTH} \
        --use_bias ${BIAS} \
        --glu_type ${ACT_TYPE} \
        --n_experts ${N_EXP} \
        --top_k ${TOPK} \
        --keeptok ${KEEPTOP} \
        --router_noise_std 0.1 \
        --sparse_compute True \
        --lb_coef 1e-2 \
        --cov_coef 1e-3 \
        --ffn_norm rmsnorm \
        --moe_dropout 0.0 \
        --moe_resid_dropout 0.0 \
        --prenorm_router true \
        --moe_aux_coef 1.0 \
        --concept none \
        --net_type ${NET_TYPE}
done
