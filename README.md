# GOPT: Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment
 - [News](#News)
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [Data](#Data)
 - [Train and evaluate GOPT with speechocean 762 dataset](#Train-and-evaluate-GOPT-with-speechocean-762-dataset)
 - [Pretrained Models](#Pretrained-Models)
 - [Test Your Own Model with Our Speechocean762 Traning Pipeline](#Test-Your-Own-Model-with-Our-Speechocean762-Traning-Pipeline)
 - [Contact](#Contact)

## News

January 2023, [YIFAN WANG](https://github.com/LyWangPX) contributed a step-by-step tutorial on how to do inference with your own data, see [[Here]](https://github.com/YuanGongND/gopt/blob/master/steps_of_inference.md). However, a [bug](https://github.com/YuanGongND/gopt/issues/15) has been reported and needs to be addressed before use.

## Introduction  

<p align="center"><img src="https://raw.githubusercontent.com/YuanGongND/gopt/master/figure/gopt_poster.png?token=GHSAT0AAAAAABHNWFG3TQB7DLK6ABX2IBSEYTJSGHA" alt="Illustration of GOPT." width="800"/></p>
<p align="center"><img src="https://github.com/YuanGongND/gopt/blob/master/figure/gopt_rev.png?raw=true" alt="Illustration of GOPT." width="800"/></p>

This repository contains the official implementation and pretrained model (in PyTorch) of the **Goodness Of Pronunciation Feature-Based Transformer (GOPT)** proposed in the ICASSP 2022 paper [Transformer-Based Multi-Aspect Multi-Granularity Non-native English Speaker Pronunciation Assessment](https://arxiv.org/abs/2205.03432) (Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang, James Glass; MIT & PAII).  

GOPT is the first model to simultaneously consider **multiple** pronunciation quality aspects (accuracy, fluency, prosody, etc) along with **multiple** granularities (phoneme, word, utterance). With a public automatic speech recognition (ASR) model, it achieves ``0.612`` phone-level Pearson correlation coefficient (PCC), ``0.549`` word-level PCC, and ``0.742`` sentence-level PCC, all are the best results on SpeechOcean762. [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-based-multi-aspect-multi/phone-level-pronunciation-scoring-on)](https://paperswithcode.com/sota/phone-level-pronunciation-scoring-on?p=transformer-based-multi-aspect-multi) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-based-multi-aspect-multi/word-level-pronunciation-scoring-on)](https://paperswithcode.com/sota/word-level-pronunciation-scoring-on?p=transformer-based-multi-aspect-multi) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-based-multi-aspect-multi/utterance-level-pronounciation-scoring-on)](https://paperswithcode.com/sota/utterance-level-pronounciation-scoring-on?p=transformer-based-multi-aspect-multi)

We intend to make our results easy to reproduce, specifically, we provide our Kaldi intermediate outputs so that you can reproduce our result **without** Kaldi in almost one-click (just download our Kaldi output and `./run.sh`, or even more simpler, run the Google Colab script [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuanGongND/gopt/blob/master/colab/GOPT_GPU.ipynb)).


## Citing  
Please cite our paper if you find this repository useful.

```
@INPROCEEDINGS{gong_gopt,
  author={Gong, Yuan and Chen, Ziyi and Chu, Iek-Heng and Chang, Peng and Glass, James},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment}, 
  year={2022},
  pages={7262-7266},
  doi={10.1109/ICASSP43922.2022.9746743}}
```

## Data

The SpeechOcean762 dataset used in ths paper is an open dataset licenced with CC BY 4.0. It can be downloaded from [this link](https://www.openslr.org/101).

## (Google Colab Scripts) Train and evaluate GOPT with speechocean 762 dataset

We provide Google Colab Script [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuanGongND/gopt/blob/master/colab/GOPT_GPU.ipynb) for quick test. 

**What you need:** 
- A free google account.
  - A (paid) Google Colab Pro plan could speed up training, but is not necessary. Free version can run the script, just a bit slower.

**What you don't need:** 
- Download GOP features manually (The Colab script download it to cloud automatically)
- GPU or any other hardware (Google Colab provides free GPUs)
- Any enviroment setting and package installation (Google Colab provides a ready-to-use environment)
- A specific operating system (You only need a web browser, e.g., Chrome)

**Please Note**
- This script is slightly different with our local script, thus the results might also be slightly different, though in our test, they are very close.
- This script does not extract Kaldi GOP features, instead, it loads our Kaldi GOP features. If you want to extract the feature by yourself, see local scripts below.

## (Local Scripts) Train and evaluate GOPT with speechocean 762 dataset

The following is a step-by-step instruction of training and evaluating GOPT with the speechocean 762 dataset. 

If you are not familiar with Kaldi, or you are not interested in GOPT feature generation, we provide our intermediate GOP features and this recipe is **Kaldi-Free** (please see below for details).
Otherwise if you want to use your own ASR model, you can NOT skip step 1 and 2.

**Step 0. Prepare the environment.**

Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
# use absolute path of this repo
gopt_path=your_gopt_path
cd $gopt_path
python3 -m venv venv-gopt
source venv-gopt/bin/activate
pip install -r requirements.txt 
```

**Step 1. Prepare the speechocean762 dataset and generate the Godness of Pronunciation (GOP) features.**

*(This step is Kaldi dependent and require familiarity with Kaldi. You can skip this step and step 2 by using our output of this step (download via the [dropbox link](https://www.dropbox.com/s/zc6o1d8rqq28vci/data.zip?dl=1) or [腾讯微云链接](https://share.weiyun.com/vJCAXjFY), please see [[here]](https://github.com/YuanGongND/gopt/tree/master/data) for details.))*

Downlod the [speechocean762](https://arxiv.org/abs/2104.01378) dataset from [[here]](https://www.openslr.org/101/). Use your own Kaldi ASR model or public Kaldi ASR model (e.g., the [Librispeech ASR Chain Model](https://kaldi-asr.org/models/m13) we used) and run [Kaldi GOP recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/gop_speechocean762) following its instruction. After the run finishes, you should see the performance of the baseline model with the ASR model you use.

Then, extract the GOP features from the intermediate files of the Kaldi GOP recipe run. 

```
kaldi_path=your_kaldi_path
cd $gopt_path
mkdir -p data/raw_kaldi_gop/librispeech
cp src/extract_kaldi_gop/{extract_gop_feats.py,extract_gop_feats_word.py} ${kaldi_path}/egs/gop_speechocean762/s5/local/
cd ${kaldi_path}/egs/gop_speechocean762/s5
python local/extract_gop_feats.py
python local/extract_gop_feats_word.py
cd $gopt_path
cp -r ${kaldi_path}/egs/gop_speechocean762/s5/gopt_feats/* data/raw_kaldi_gop/librispeech
```

For questions regarding the Kaldi recipe (e.g., how to generate GOP feature for a single wav file), please kindly check the issues of the Kaldi GOP recipe at [[here]](https://github.com/kaldi-asr/kaldi/issues?q=is%3Aissue+gop).

**Step 2. Convert GOP features and labels to sequences**

*(You can skip this step and step 1 by using our output of this step (download via the [dropbox link](https://www.dropbox.com/s/zc6o1d8rqq28vci/data.zip?dl=1) or [腾讯微云链接](https://share.weiyun.com/vJCAXjFY), please see [[here]](https://github.com/YuanGongND/gopt/tree/master/data) for details.))*

The Kaldi output GOP features and labels are at phone level. To model pronunciation assessment as a sequence-to-sequence problem, we need to convert the feature to shape like ``[#utterance, seq_len, feat_dim]``. 
Specifically, we pad all utterance into 50 tokens (phones) with -1, i.e., ``seq_len=50``. The padded tokens are masked out for any metric calculation. 

Use the following scripts for this step:
```
mkdir data/seq_data_librispeech
cd src/prep_data
python gen_seq_data_phn.py
python gen_seq_data_word.py
python gen_seq_data_utt.py
```

**Step 3. Run Training and Evaluation**

The entry point of the training and evaluation scripts is ``gopt/src/run.sh``, which calls ``gopt/src/traintest.py``, which then calls ``gopt/src/models/gopt.py``.
Just run the following code snippet.

```
cd gopt/src
(slurm user) sbatch run.sh
(local user) ./run.sh
```
Results, best model, and predictions will be saved in the ``exp_dir`` specified in ``gopt/src/run.sh``.

## Pretrained Models
We provide three pretrained models and corresponding training logs. They are in ``gopt/pretrained_models/``.

- **Pretrained GOPT Models**: We provide three pretrained GOPT models trained with various GOP features. These models generally perform better than the results reported in the paper because we report mean result of 5 runs with different random seeds in the paper while release the best model.

|                    | Phn MSE | Phn PCC | Word Acc PCC | Word Str PCC | Word Total PCC | Utt Acc PCC | Utt Comp PCC | Utt Flu PCC | Utt Pros PCC | Utt Total PCC |
|--------------------|:-------:|:-------:|:------------:|:------------:|:--------------:|:-----------:|:------------:|:-----------:|:------------:|:-------------:|
| GOPT (Librispeech) |  0.084  |  0.616  |     0.536    |     0.326    |      0.552     |    0.718    |     0.109    |    0.756    |     0.764    |     0.743     |
| GOPT (PAII-A)      |  0.069  |  0.679  |     0.595    |     0.150    |      0.606     |    0.727    |    -0.044    |    0.692    |     0.695    |     0.731     |
| GOPT (PAII-B)      |  0.071  |  0.664  |     0.592    |     0.174    |      0.602     |    0.722    |     0.122    |    0.721    |     0.723    |     0.740     |

- **Training Logs**: Training logs are in ``gopt_{librispeech,paiia,paiib}/result.csv`` in shape  ``[num_epoch, #metrics]`` where there are in total 32 columns: column ``[0]`` is the epoch id, ``[1-4]`` are phone-level training mse, training pcc, test mse, test pcc, respectively; ``[5]`` is the learning rate of the epoch;
``[6-10, 11-15, 16-20, 21-25]`` are utterance-level training mse, training pcc, test mse, test pcc, respectively, each contains 5 scores of ``accuracy, completeness, fluency, prosodic, total``.
``[26-28, 29-31]`` are word-level training pcc and test pcc, respectively, each contains 3 scores of ``accuracy, stress, total``. 

- **Acoustic Models**: Librispeech acoustic model is publicly available at https://kaldi-asr.org/models/m13. PAII acoustic models will not be released.

## Test Your Own Model with Our Speechocean762 Traning Pipeline
It is extremely easy to train and test your model with our speechocean762 training pipeline and compare it with GOPT. You don't even need Kaldi or any data processing if you plan to use the same ASR models with us. 

Specifically, your model need to be in ``pytorch`` and take input and generate output in the following format:

- Input: GOP feature ``x`` in shape ``[batch_size, seq_len, feat_dim]``, e.g., ``[25, 50, 84]`` for a batch of 25 utterances, each with 50 phones after -1 padding, and each phone has a GOP feature vector of dimension 84. Note the GOP feature dimension varies with the ASR model, so your model should be able to process various ``feat_dim``.
- (Optional) Input: canonical phone``phn`` in shape ``[batch_size, seq_len, phn_num]``, e.g., ``[25, 50, 40]`` for a batch of 25 utterance, each with 50 phones after padding with a phone dictionary of size of 40. For speechocean762, ``phn_num=40``.
- Output: Tuple of ``[u1, u2, u3, u4, u5, p, w1, w2, w3]`` where ``u{1-5}`` are utterance-level scores in shape ``[batch_size, 1]``; ``p`` and ``w{1-3}`` are phone-level and word-level score in shape ``[batch_size, seq_len]``. Note we propagate word score to phone-level, so word output should also be at phone-level.  

Add your model to ``gopt/src/models/``, modify ``gopt/src/models/__init__.py`` and ``gopt/src/traintest.py`` to include your model. Then just follow the [instructions](#Train-and-evaluate-GOPT-with-speechocean-762-dataset). You can skip step 1 and 2 by using our intermediate data files. 

 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu. For questions regarding the Kaldi recipe (e.g., how to generate GOP feature for a single wav file), please kindly check the issues of the Kaldi GOP recipe at [[here]](https://github.com/kaldi-asr/kaldi/issues?q=is%3Aissue+gop).

