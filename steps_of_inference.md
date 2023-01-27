# Infer your own data

This tutorial is aimed at helping people understand how to infer any data with GOPT.
There will be some restrictions on "any data", I will explain it later.

1. Install Kaldi and GOPT. Sometimes Kaldi can be fuzzy. If meeting any problems, use a docker image of Kaldi instead.
2. Download the original `speechocean762` to your disk.
3. Copy `speechocean762` and change to your dataset’s name. In this example, I use `test_dataset`
4. There are multiple hacks we need to do to make our `test_dataset` runnable.
5. First, delete unnecessary .wav files. Replace them with your own.
6. Update all files in both test and train. Including `spk2age`, `spk2gender`, `spk2utt`, `text`, `utt2spk`, `wav.scp`. Remember, words in text need to be capitalized.
7. Because I only have 1 wav file, all these files will be 1-line. For example, I set the speaker ID to be `0001` and utt_id to be `test`. So:
    
    `wav.scp` includes
    
    ```
    test	WAVE/SPEAKER0001/test.wav
    ```
    
    `spk2utt` includes
    
    ```
    0001 test
    ```
    
    You can infer how to write the rest.
    
8. In `resource/text-phone`, delete unnecessary lines and replace your own. Here, each lines begins with \<utt_id\>.\<n\> which represents the n-th word in your text. After it, please append the corresponding phones of that word. 
    
    To be specific, find your corresponding phones in `resource/lexicon.txt`. For instance, the word FAN would be `F AE0 N`. For all the first phones, add an additional suffix B, for the last phones, add an additional suffix E and for all others, add the suffix I.
    
    For example, if my text is “FAN WORKS”, the final result in text-phone is
    
    ```
    test.0 F_B AE0_I N_E
    test.1 W_B ER0_I K_I S_E
    ```
    
    If your don’t do this right, you will stuck at stage 8.
    
9. Download and extract all tars in [https://kaldi-asr.org/models/m13](https://kaldi-asr.org/models/m13)
10. In `gop_speechocean762/s5/run.sh` , change lines 38-42 to your extracted results. In my case, I use
    
    ```
    librispeech_eg=../../librispeech/s5
    model=$librispeech_eg/exp/chain_cleaned/tdnn_1d_sp
    ivector_extractor=$librispeech_eg/exp/nnet3_cleaned/extractor
    lang=$librispeech_eg/data/lang_test_tgsmall
    ```
    
    Also, change `stage` to `2`to avoid download. Change `nj` to the number of examples. In my case, I set to 1.
    
11. After running run.sh, confirm you have non-empty feat files in `gop_train` and `gop_test`. Mine looks like this 
    
    ```
    (gopt) yifan@XXX:~/develop/kaldi/egs/gop_speechocean762/s5/exp/gop_train$ tree .
    .
    ├── feat.1.ark
    ├── feat.1.scp
    ├── feat.scp
    ├── gop.1.ark
    ├── gop.1.scp
    ├── gop.scp
    └── log
        └── compute_gop.1.log
    ```
    
12. Execute the following original guide in GOPT.
    
    ```
    kaldi_path=your_kaldi_path
    cd $gopt_path
    mkdir -p data/raw_kaldi_gop/librispeech
    cp src/extract_kaldi_gop/{extract_gop_feats.py,extract_gop_feats_word.py} ${kaldi_path}/egs/gop_speechocean762/s5/local/
    cd ${kaldi_path}/egs/gop_speechocean762/s5
    ```
    
13. Now we need to change the original GOPT files
14. First, in `extract_gop_feats.py`, delete the continue [https://github.com/YuanGongND/gopt/blob/master/src/extract_kaldi_gop/extract_gop_feats.py#L54](https://github.com/YuanGongND/gopt/blob/master/src/extract_kaldi_gop/extract_gop_feats.py#L54). (PS: `label` in this file is `lable`. Ummm, you can’t unsee them.)
15. In the same file, because we do not have score, change [https://github.com/YuanGongND/gopt/blob/master/src/extract_kaldi_gop/extract_gop_feats.py#L61](https://github.com/YuanGongND/gopt/blob/master/src/extract_kaldi_gop/extract_gop_feats.py#L61) to
    
    ```
    lables.append([ph])
    ```
    
16. Run the edited `python local/extract_gop_feats.py`, skip `extract_gop_feats_word.py`, not needed for inference.
17. Continue with 
    
    ```
    cd $gopt_path
    cp -r ${kaldi_path}/egs/gop_speechocean762/s5/gopt_feats/* data/raw_kaldi_gop/<your dataset name>
    ```
    
18. Change another GOPT file, `src/prep_data/gen_seq_data_phn.py`. Because we do not have score any more, all we want to have is the phn. Also we need to replace the hardcoding path to \<your dataset name\>. You can debug it yourself, here is my edited results.
    
    ```
    # -*- coding: utf-8 -*-
    # @Time    : 9/19/21 11:13 PM
    # @Author  : Yuan Gong
    # @Affiliation  : Massachusetts Institute of Technology
    # @Email   : yuangong@mit.edu
    # @File    : gen_seq_data_phn.py
    
    # Generate sequence phone input and label for seq2seq models from raw Kaldi GOP features.
    
    import numpy as np
    
    def load_feat(path):
        file = np.loadtxt(path, delimiter=',')
        return file
    
    def load_keys(path):
        file = np.loadtxt(path, delimiter=',', dtype=str)
        return file
    
    def load_label(path):
        file = np.loadtxt(path, delimiter=',', dtype=str)
        return file
    
    def process_label(label):
        pure_label = []
        for i in range(0, label.shape[0]):
            pure_label.append(float(label[i, 1]))
        return np.array(pure_label)
    
    def process_feat_seq(feat, keys, labels, phn_dict):
        key_set = []
        for i in range(keys.shape[0]):
            cur_key = keys[i].split('.')[0]
            key_set.append(cur_key)
    
        feat_dim = feat.shape[1] - 1
    
        utt_cnt = len(list(set(key_set)))
        print('In total utterance number : ' + str(utt_cnt))
    
        # Pad all sequence to 50 because the longest sequence of the so762 dataset is shorter than 50.
        seq_feat = np.zeros([utt_cnt, 50, feat_dim])
        # -1 means n/a, padded token
        # [utt, seq_len, 0] is the phone label, and the [utt, seq_len, 1] is the score label
        seq_label = np.zeros([utt_cnt, 50, 2]) - 1
    
        # the key format is utt_id.phn_id
        prev_utt_id = keys[0].split('.')[0]
    
        row = 0
        for i in range(feat.shape[0]):
            cur_utt_id, cur_tok_id = keys[i].split('.')[0], int(keys[i].split('.')[1])
            # if a new sequence, start a new row of the feature vector.
            if cur_utt_id != prev_utt_id:
                row += 1
                prev_utt_id = cur_utt_id
    
            # The first element is the phone label.
            seq_feat[row, cur_tok_id, :] = feat[i, 1:]
    
            # [utt, seq_len, 0] is the phone label
            print(labels)
            seq_label[row, cur_tok_id, 0] = phn_dict[labels[i]]
            # [utt, seq_len, 1] is the score label, range from 0-2
            # seq_label[row, cur_tok_id, 1] = labels[i, 1]
    
        return seq_feat, seq_label
    
    def gen_phn_dict(label):
        phn_dict = {}
        phn_idx = 0
        for i in range(label.shape[0]):
            if label[i] not in phn_dict:
                phn_dict[label[i]] = phn_idx
                phn_idx += 1
        return phn_dict
    
    # generate sequence training data
    tr_feat = load_feat('../../data/raw_kaldi_gop/test_dataset/tr_feats.csv')
    tr_keys = load_keys('../../data/raw_kaldi_gop/test_dataset/tr_keys_phn.csv')
    tr_label = load_label('../../data/raw_kaldi_gop/test_dataset/tr_labels_phn.csv')
    phn_dict = gen_phn_dict(tr_label)
    print(phn_dict)
    tr_feat, tr_label = process_feat_seq(tr_feat, tr_keys, tr_label, phn_dict)
    print(tr_feat.shape)
    print(tr_label.shape)
    np.save('../../data/seq_data_test_dataset/tr_feat.npy', tr_feat)
    np.save('../../data/seq_data_test_dataset/tr_label_phn.npy', tr_label)
    
    # generate sequence test data
    te_feat = load_feat('../../data/raw_kaldi_gop/test_dataset/te_feats.csv')
    te_keys = load_keys('../../data/raw_kaldi_gop/test_dataset/te_keys_phn.csv')
    te_label = load_label('../../data/raw_kaldi_gop/test_dataset/te_labels_phn.csv')
    te_feat, te_label = process_feat_seq(te_feat, te_keys, te_label, phn_dict)
    print(te_feat.shape)
    print(te_label.shape)
    np.save('../../data/seq_data_test_dataset/te_feat.npy', te_feat)
    np.save('../../data/seq_data_test_dataset/te_label_phn.npy', te_label)
    ```
    
19. The last step requires you to run these lines. Skip word and utterence.

    ```
    mkdir data/seq_data_<your dataset name>
    cd src/prep_data
    python gen_seq_data_phn.py
    ```

20. Finally, in `gopt/data/<your dataset name>`, you will see kindly two files that are needed to do the inference. `te_feat.npy` and `te_label_phn.npy`. But remember, the `te_label_phn.npy`contains both `phn` and `scores` which we have not generated (and not needed). So, in order to do the inference, run the following.
    
    PS: to simplify stuff, my train dataset is the same as test dataset.
    
    ```python
    
    import torch
    import sys
    import os
    sys.path.append(os.path.abspath('../src/'))
    from models import GOPT
    gopt = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    # GOPT is trained with dataparallel, so it need to be wrapped with dataparallel even you have a single gpu or cpu
    gopt = torch.nn.DataParallel(gopt)
    sd = torch.load('gopt_librispeech/best_audio_model.pth', map_location='cpu')
    gopt.load_state_dict(sd, strict=True)
    
    import numpy as np
    input_feat = np.load("<your_path>/te_feat.npy")
    input_phn = np.load("<your_path>/te_label_phn.npy")
    t_input_feat = torch.from_numpy(input_feat)
    t_phn = torch.from_numpy(input_phn[:,:,0])
    gopt = gopt.float()
    gopt.eval()
    with torch.no_grad():
        print(gopt(t_input_feat.float(),t_phn.float()))
    ```
    

Good Luck!
Restrictions: If your text contains words that are not in lexicon, you are out of luck.