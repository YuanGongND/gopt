- **Pretrained GOPT Models**: We provide three pretrained GOPT models trained with various GOP features. These models generally perform better than the results reported in the paper because we report mean result of 5 runs with different random seeds in the paper while release the best model.

    |                    | Phn MSE | Phn PCC | Word Acc PCC | Word Str PCC | Word Total PCC | Utt Acc PCC | Utt Comp PCC | Utt Flu PCC | Utt Pros PCC | Utt Total PCC |
    |--------------------|:-------:|:-------:|:------------:|:------------:|:--------------:|:-----------:|:------------:|:-----------:|:------------:|:-------------:|
    | GOPT (Librispeech) |  0.084  |  0.616  |     0.536    |     0.326    |      0.552     |    0.718    |     0.109    |    0.756    |     0.764    |     0.743     |
    | GOPT (PAII-A)      |  0.069  |  0.679  |     0.595    |     0.150    |      0.606     |    0.727    |    -0.044    |    0.692    |     0.695    |     0.731     |
    | GOPT (PAII-B)      |  0.071  |  0.664  |     0.592    |     0.174    |      0.602     |    0.722    |     0.122    |    0.721    |     0.723    |     0.740     |

- **Training Logs**: Training logs are in ``gopt_{librispeech,paiia,paiib}/result.csv`` in shape  ``[num_epoch, #metrics]`` where there are in total 31 metrics:  ``[1-4]`` are phone-level training mse, training pcc, test mse, test pcc, respectively; ``[5]`` is the learning rate of the epoch;
``[6-10, 11-15, 16-20, 21-25]`` are utterance-level training mse, training pcc, test mse, test pcc, respectively, where each contains 5 scores of ``accuracy, completeness, fluency, prosodic, total``.
``[26-28, 29-31]`` are word-level training pcc and test pcc, respectively, where each contains 3 scores of ``accuracy, stress, total``.

- **Acoustic Models**: Librispeech acoustic model is publicly available at https://kaldi-asr.org/models/m13. PAII acoustic models will not be released.