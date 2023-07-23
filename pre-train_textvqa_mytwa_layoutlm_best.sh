# set -ex
# export CUDA_DEVICE_ORDER='PCI_BUS_ID'
# export CUDA_VISIBLE_DEVICES=1
export NCCL_TIMEOUT=3600000
python -m torch.distributed.launch --nproc_per_node 2 tools/run.py --pretrain --tasks vqa \
    --datasets m4c_textvqa --model mytwa --seed 13 \
    --config configs/vqa/m4c_textvqa/twa_layout_base_adv_pretrain.yml \
    --save_dir save/m4c_split_pretrain_textvqa_mytwa_layout0720 training_parameters.distributed True