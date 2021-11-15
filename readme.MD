# GOPT: Transformer-Based Multi-Aspect Multi-Granularity Non-native English Speaker Pronunciation Assessment.
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

Step 1. Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies.


Step 2. Test the AST model.

## Pretrained Models
We provide full AudioSet pretrained models.

 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.
