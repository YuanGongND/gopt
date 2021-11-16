# GOPT: Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [Train and evaluate GOPT with speechocean 762 dataset](#Train-and-evaluate-GOPT-with-speechocean-762-dataset)
 - [Pretrained Models](#Pretrained-Models)
 - [Test Your Own Model with Our Speechocean762 Traning Pipeline](#Test-Your-Own-Model-with-Our-Speechocean762-Traning-Pipeline)
 - [Contact](#Contact)

## Introduction  

<p align="center"><img src="https://raw.githubusercontent.com/YuanGongND/gopt/master/figure/gopt.png?token=AEC6JZXNM2T3SJNHAIZ5QW3BTPH6I" alt="Illustration of AST." width="800"/></p>

This repository contains the official implementation and pretrained model (in PyTorch) of the **Goodness Of Pronunciation Feature-Based Transformer (GOPT)** proposed in the ICASSP 2022 paper [Transformer-Based Multi-Aspect Multi-Granularity Non-native English Speaker Pronunciation Assessment](https://arxiv.org/abs/dummy) (Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang, James Glass).  

GOPT is the first model to simultaneously consider *multiple* pronunciation quality aspects (accuracy, fluency, prosody, etc) along with *multiple* granularities (phoneme, word, utterance). With a public automatic speech recognition (ASR) model, it achieves ``0.612`` phone-level Pearson correlation coefficient (PCC), ``0.549`` word-level PCC, and ``0.742`` sentence-level PCC.


## Citing  
Please cite our paper if you find this repository useful.

TBD
  
## Train and evaluate GOPT with speechocean 762 dataset

The following is a step-by-step instruction of training and evaluating GOPT with the speechocean 762 dataset. If you are only interest in 
If you want to use your own ASR model, you cannot skip step 1 and 2.

**Step 0. Prepare the environment.**

Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.

```
cd gopt/ 
python3 -m venv venv-gopt
source venv-gopt/bin/activate
pip install -r requirements.txt 
```

**Step 1. Prepare the speechocean762 dataset and generate the Godness of Pronunciation (GOP) features.**

(This step is Kaldi dependent and require familiarity with Kaldi. You can skip this step by using our output of this step, which can be downloaded [here]().)

Downlod the [speechocean762](https://arxiv.org/abs/2104.01378) dataset from [here](https://www.openslr.org/101/). Use your own Kaldi ASR model or public Kaldi ASR model (e.g., the [Librispeech ASR Chain Model](https://kaldi-asr.org/models/m13) we used) and run [Kaldi GOP recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/gop_speechocean762). After the run finishes, you should see the performance of the baseline model with the ASR model you use.

Then, extract the GOP features from the intermediate files of the Kaldi recipe run. Put ``load_gop_feats.py`` and ``load_gop_feats_word.py`` in ``your_kaldi_path/egs/speechocean762/s5/local/``. Go back to (``cd ../``) to ``/speechocean762/s5/``, and run both scripts by ``python local/load_gop_feats.py`` and ``python local/load_gop_feats_word.py``. 
You can use the same python environment as the recipe (our environment is in requirement.txt). ’load_gop_feats.py’ script will save 12 files in /speechocean762/s5/yuan_gop_feats/ and ‘load_gop_feats_word.py’ will save 6 files; please send me these files (if you have 3 AMs, each will have 12+6 outputs, and in total 54 files).

**Step 2. Convert Kaldi Output to Se**

The output GOP features are at phone level. To model pronunciation assessment as a sequence-to-sequence problem, we need to convert to 


**Step 2. Test the AST model**

## Pretrained Models
We provide three pretrained models and corresponding training logs. They are in ``gopt/pretrained_models/``.

- **Pretrained GOPT Models**: We provide three pretrained GOPT models trained with various GOP features. These models generally perform better than the results reported in the paper because we report mean result of 5 runs with different random seeds in the paper while release the best model.

|                    | Phn MSE | Phn PCC | Word Acc PCC | Word Str PCC | Word Total PCC | Utt Acc PCC | Utt Comp PCC | Utt Flu PCC | Utt Pros PCC | Utt Total PCC |
|--------------------|:-------:|:-------:|:------------:|:------------:|:--------------:|:-----------:|:------------:|:-----------:|:------------:|:-------------:|
| GOPT (Librispeech) |  0.084  |  0.616  |     0.536    |     0.326    |      0.552     |    0.718    |     0.109    |    0.756    |     0.764    |     0.743     |
| GOPT (PAII-A)      |  0.069  |  0.679  |     0.595    |     0.150    |      0.606     |    0.727    |    -0.044    |    0.692    |     0.695    |     0.731     |
| GOPT (PAII-B)      |  0.071  |  0.664  |     0.592    |     0.174    |      0.602     |    0.722    |     0.122    |    0.721    |     0.723    |     0.740     |

- **Training Logs**: Training logs are in ``gopt_{librispeech,paiia,paiib}/result.csv`` in shape  ``[num_epoch, #metrics]`` where there are in total 31 metrics:  ``[1-4]`` are phone-level training mse, training pcc, test mse, test pcc, respectively; ``[5]`` is the learning rate of the epoch;
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
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.
