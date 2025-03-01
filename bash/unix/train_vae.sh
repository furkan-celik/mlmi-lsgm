python train_vae.py --data ../data/raw/kaggle/kermany2018/OCT2017/ --root MNIST/ --save save_dir/vae --dataset retina \
      --resize 256 --crop 256 \
      --batch_size 2 --epochs 200 --num_latent_scales 1 --num_groups_per_scale 2 --num_postprocess_cells 3 \
      --num_preprocess_cells 3 --num_cell_per_cond_enc 1 --num_cell_per_cond_dec 1 --num_latent_per_group 20 \
      --num_preprocess_blocks 2 --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 64 \
      --num_channels_dec 64 --decoder_dist bin --kl_anneal_portion 1.0 --kl_max_coeff 0.7 --channel_mult 1 2 2 \
      --num_nf 0 --arch_instance res_mbconv --num_process_per_node 1 --use_se