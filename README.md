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