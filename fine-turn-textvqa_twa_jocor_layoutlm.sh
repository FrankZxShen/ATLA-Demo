# set -ex
# export CUDA_DEVICE_ORDER='PCI_BUS_ID'
# export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_IB_DISABLE=1
# export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=3600000
python -m torch.distributed.launch --nproc_per_node 2  tools/run.py \
       --tasks vqa --datasets m4c_textvqa --model mytwa --seed 13 \
       --config configs/vqa/m4c_textvqa/twa_layout_refine.yml --save_dir save/m4c_split_refine_textvqa_twa_layout0723 \
       --resume_file save/m4c_split_pretrain_textvqa_mytwa_layout0720/m4c_textvqa_mytwa_13/best.ckpt training_parameters.distributed True