<details><summary>Download checkpoints</summary>

1. Download checkpoint.pt from https://drive.google.com/file/d/1U9bVcLXmjtP1XW8gLwSG4-ILgvdedabK/view?usp=sharing
Put to:
```shell script
mlmi-lsgm/MNIST/
```

2. Download checkpoint_nll.pt from https://drive.google.com/file/d/1NEbX-nSWDixtS8LoB-RmyKNW8td_aXd1/view?usp=sharing
Put to:
```shell script
mlmi-lsgm/MNIST/vada
```

3. Download checkpoint.pt from https://drive.google.com/file/d/1zMAE9S0AmDLL8P8Qh5a8JTZGQFsbfu60/view?usp=sharing
Put to:
```shell script
mlmi-lsgm/save_dir/vae
```
</details>

<details><summary>Setup environment</summary>

```shell script
cd mlmi-lsgm
pip install -r requirements.txt
pip install blobfile
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
</details>

<details><summary>Train VAE</summary>

```shell script
bash bash/unix/train_vae.sh
```
</details>

<details><summary>Train LSGM (using pretrained VAE)</summary>

```shell script
bash bash/unix/train_vada_from_vae.sh
```
</details>

<details><summary>Train LSGM from existing checkpoint </summary>

```shell script
bash bash/unix/train_vada_from_vada.sh
```
</details>