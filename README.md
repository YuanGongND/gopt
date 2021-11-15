# GOPT: Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment
 - [Introduction](#Introduction)
 - [Citing](#Citing)  
 - [Run GOPT](#Getting-Started)
 - [Pretrained Models](#Pretrained-Models)
 - [Contact](#Contact)

## Introduction  

<p align="center"><img src="https://raw.githubusercontent.com/YuanGongND/gopt/master/figure/gopt.png?token=AEC6JZXNM2T3SJNHAIZ5QW3BTPH6I" alt="Illustration of AST." width="800"/></p>

This repository contains the official implementation and pretrained model (in PyTorch) of the **Goodness Of Pronunciation Feature-Based Transformer (GOPT)** proposed in the ICASSP 2022 paper [Transformer-Based Multi-Aspect Multi-Granularity Non-native English Speaker Pronunciation Assessment](https://arxiv.org/abs/dummy) (Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang, James Glass).  

GOPT is the first model to simultaneously consider *multiple* pronunciation quality aspects (accuracy, fluency, prosody, etc) along with *multiple* granularities (phoneme, word, utterance). With a public automatic speech recognition (ASR) model, it achieves ``0.612`` phone-level Pearson correlation coefficient (PCC), ``0.549`` word-level PCC, and ``0.742`` sentence-level PCC.


## Citing  
Please cite our paper if you find this repository useful.

TBD
  
## Run GOPT

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


**Step 2. Test the AST model**

## Pretrained Models
We provide full AudioSet pretrained models.

 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.
