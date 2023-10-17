## Beware of Model Collapse! Fast and Stable Test-time Adaptation for Robust Question Answering

This is the official project repository for [Beware of Model Collapse! Fast and Stable Test-time Adaptation for Robust Question Answering](https://openreview.net/forum?id=BSApuhuM87) (EMNLP 2023, Main).

Anti-CF utilizes the output of the source model as a soft label to regularize the update of the adapted model during test time to ensure that the adapted model will not deviate too far from the source model, thus avoiding model collapse.  

To reduce the inference time, we freeze the source model and add an efficient side block as the adapted model to reduce the cost of additional forward propagation and back propagation.

## Quick start

#### Environment

You should run the following script to install the dependencies first.

```
conda create -n AntiCF python==3.8
conda activate AntiCF
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==2.4.1
```

#### Download the data

You can download the datasets from [here](https://huggingface.co/datasets/virtuous/ColdQA_warmup).

#### Obtain the Base Model

you can train your own model with [xTreme](https://github.com/google-research/xtreme) or [xTune](https://github.com/bozheng-hit/xTune).

#### Run baselines

You can run all baselines in the paper by:

```
python baselines.py --device {device} --model_path {model_path} --method {method} --output_path {output_path}
```

#### Run Anti-CF

You can run Anti-CF by:

```
python AntiCF.py --device {device} --model_path {model_path} --alpha {alpha} --output_path {output_path}
```



