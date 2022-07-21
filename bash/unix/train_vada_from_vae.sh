python train_vada.py --data ../data/raw/kaggle/kermany2018/OCT2017/ --resize 256 --crop 256 --root MNIST \
        --save vada --dataset retina --epochs 800 \
        --dropout 0.2 --batch_size 4 --num_scales_dae 2 --weight_decay_norm_vae 1e-2 \
        --weight_decay_norm_dae 1e-2 --num_channels_dae 256 --train_vae  --num_cell_per_scale_dae 1 \
        --learning_rate_dae 3e-4 --learning_rate_min_dae 3e-4 --train_ode_solver_tol 1e-5 --cont_kl_anneal  \
        --sde_type vpsde --iw_sample_p ll_iw --num_process_per_node 1 --use_se \
        --vae_checkpoint save_dir/vae/checkpoint_epoch_100.pt  --dae_arch ncsnpp --embedding_scale 1000 \
        --mixing_logit_init -6 --warmup_epochs 20 --drop_inactive_var --skip_final_eval --fid_dir save_dir/fids