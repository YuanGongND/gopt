# -*- coding: utf-8 -*-
# @Time    : 9/20/21 12:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# train and test the models
import sys
import os
import time
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from models import *
import argparse

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, default="./exp/", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n-epochs", type=int, default=100, help="number of maximum training epochs")
parser.add_argument("--goptdepth", type=int, default=1, help="depth of gopt models")
parser.add_argument("--goptheads", type=int, default=1, help="heads of gopt models")
parser.add_argument("--batch_size", type=int, default=25, help="training batch size")
parser.add_argument("--embed_dim", type=int, default=12, help="gopt transformer embedding dimension")
parser.add_argument("--loss_w_phn", type=float, default=1, help="weight for phoneme-level loss")
parser.add_argument("--loss_w_word", type=float, default=1, help="weight for word-level loss")
parser.add_argument("--loss_w_utt", type=float, default=1, help="weight for utterance-level loss")
parser.add_argument("--model", type=str, default='gopt', help="name of the model")
parser.add_argument("--am", type=str, default='librispeech', help="name of the acoustic models")
parser.add_argument("--noise", type=float, default=0., help="the scale of random noise added on the input GoP feature")

# just to generate the header for the result.csv
def gen_result_header():
    phn_header = ['epoch', 'phone_train_mse', 'phone_train_pcc', 'phone_test_mse', 'phone_test_pcc', 'learning rate']
    utt_header_set = ['utt_train_mse', 'utt_train_pcc', 'utt_test_mse', 'utt_test_pcc']
    utt_header_score = ['accuracy', 'completeness', 'fluency', 'prosodic', 'total']
    word_header_set = ['word_train_pcc', 'word_test_pcc']
    word_header_score = ['accuracy', 'stress', 'total']
    utt_header, word_header = [], []
    for dset in utt_header_set:
        utt_header = utt_header + [dset+'_'+x for x in utt_header_score]
    for dset in word_header_set:
        word_header = word_header + [dset+'_'+x for x in word_header_score]
    header = phn_header + utt_header + word_header
    return header

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_mse = 0, 999
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in audio_model.parameters()) / 1e3))
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(20, 100, 5)), gamma=0.5, last_epoch=-1)

    loss_fn = nn.MSELoss()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 32])

    while epoch < args.n_epochs:
        audio_model.train()
        for i, (audio_input, phn_label, phns, utt_label, word_label) in enumerate(train_loader):

            audio_input = audio_input.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)

            # warmup
            warm_up_step = 100
            if global_step <= warm_up_step and global_step % 5 == 0:
                warm_lr = (global_step / warm_up_step) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # add random noise for augmentation.
            noise = (torch.rand([audio_input.shape[0], audio_input.shape[1], audio_input.shape[2]]) - 1) * args.noise
            noise = noise.to(device, non_blocking=True)
            audio_input = audio_input + noise

            #print(phns.shape)
            u1, u2, u3, u4, u5, p, w1, w2, w3 = audio_model(audio_input, phns)

            # filter out the padded tokens, only calculate the loss based on the valid tokens
            # < 0 is a flag of padded tokens
            mask = (phn_label>=0)
            p = p.squeeze(2)
            p = p * mask
            phn_label = phn_label * mask

            loss_phn = loss_fn(p, phn_label)

            # avoid the 0 losses of the padded tokens impacting the performance
            loss_phn = loss_phn * (mask.shape[0] * mask.shape[1]) / torch.sum(mask)

            # utterance level loss, also mse
            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            loss_utt = loss_fn(utt_preds ,utt_label)

            # word level loss
            word_label = word_label[:, :, 0:3]
            mask = (word_label>=0)
            word_pred = torch.cat((w1,w2,w3), dim=2)
            word_pred = word_pred * mask
            word_label = word_label * mask
            loss_word = loss_fn(word_pred, word_label)
            loss_word = loss_word * (mask.shape[0] * mask.shape[1] * mask.shape[2]) / torch.sum(mask)

            loss = args.loss_w_phn * loss_phn + args.loss_w_utt * loss_utt + args.loss_w_word * loss_word

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('start validation')

        # ensemble results
        # don't save prediction for the training set
        tr_mse, tr_corr, tr_utt_mse, tr_utt_corr, tr_word_mse, tr_word_corr = validate(audio_model, train_loader, args, -1)
        te_mse, te_corr, te_utt_mse, te_utt_corr, te_word_mse, te_word_corr = validate(audio_model, test_loader, args, best_mse)

        print('Phone: Test MSE: {:.3f}, CORR: {:.3f}'.format(te_mse.item(), te_corr))
        print('Utterance:, ACC: {:.3f}, COM: {:.3f}, FLU: {:.3f}, PROC: {:.3f}, Total: {:.3f}'.format(te_utt_corr[0], te_utt_corr[1], te_utt_corr[2], te_utt_corr[3], te_utt_corr[4]))
        print('Word:, ACC: {:.3f}, Stress: {:.3f}, Total: {:.3f}'.format(te_word_corr[0], te_word_corr[1], te_word_corr[2]))

        result[epoch, :6] = [epoch, tr_mse, tr_corr, te_mse, te_corr, optimizer.param_groups[0]['lr']]

        result[epoch, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, te_utt_mse, te_utt_corr])

        result[epoch, 26:32] = np.concatenate([tr_word_corr, te_word_corr])

        header = ','.join(gen_result_header())
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=header, comments='')
        print('-------------------validation finished-------------------')

        if te_mse < best_mse:
            best_mse = te_mse
            best_epoch = epoch

        if best_epoch == epoch:
            if os.path.exists("%s/models/" % (exp_dir)) == False:
                os.mkdir("%s/models" % (exp_dir))
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))

        if global_step > warm_up_step:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        epoch += 1

def validate(audio_model, val_loader, args, best_mse):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_phn, A_phn_target = [], []
    A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = [], [], [], [], [], []
    A_w1, A_w2, A_w3, A_word_target = [], [], [], []
    with torch.no_grad():
        for i, (audio_input, phn_label, phns, utt_label, word_label) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            u1, u2, u3, u4, u5, p, w1, w2, w3 = audio_model(audio_input, phns)
            p = p.to('cpu').detach()
            u1, u2, u3, u4, u5 = u1.to('cpu').detach(), u2.to('cpu').detach(), u3.to('cpu').detach(), u4.to('cpu').detach(), u5.to('cpu').detach()
            w1, w2, w3 = w1.to('cpu').detach(), w2.to('cpu').detach(), w3.to('cpu').detach()

            A_phn.append(p)
            A_phn_target.append(phn_label)

            A_u1.append(u1)
            A_u2.append(u2)
            A_u3.append(u3)
            A_u4.append(u4)
            A_u5.append(u5)
            A_utt_target.append(utt_label)

            A_w1.append(w1)
            A_w2.append(w2)
            A_w3.append(w3)
            A_word_target.append(word_label)

        # phone level
        A_phn, A_phn_target  = torch.cat(A_phn), torch.cat(A_phn_target)

        # utterance level
        A_u1, A_u2, A_u3, A_u4, A_u5, A_utt_target = torch.cat(A_u1), torch.cat(A_u2), torch.cat(A_u3), torch.cat(A_u4), torch.cat(A_u5), torch.cat(A_utt_target)

        # word level
        A_w1, A_w2, A_w3, A_word_target = torch.cat(A_w1), torch.cat(A_w2), torch.cat(A_w3), torch.cat(A_word_target)

        # get the scores
        phn_mse, phn_corr = valid_phn(A_phn, A_phn_target)

        A_utt = torch.cat((A_u1, A_u2, A_u3, A_u4, A_u5), dim=1)
        utt_mse, utt_corr = valid_utt(A_utt, A_utt_target)

        A_word = torch.cat((A_w1, A_w2, A_w3), dim=2)
        word_mse, word_corr, valid_word_pred, valid_word_target = valid_word(A_word, A_word_target)

        if phn_mse < best_mse:
            print('new best phn mse {:.3f}, now saving predictions.'.format(phn_mse))

            # create the directory
            if os.path.exists(args.exp_dir + '/preds') == False:
                os.mkdir(args.exp_dir + '/preds')

            # saving the phn target, only do once
            if os.path.exists(args.exp_dir + '/preds/phn_target.npy') == False:
                np.save(args.exp_dir + '/preds/phn_target.npy', A_phn_target)
                np.save(args.exp_dir + '/preds/word_target.npy', valid_word_target)
                np.save(args.exp_dir + '/preds/utt_target.npy', A_utt_target)

            np.save(args.exp_dir + '/preds/phn_pred.npy', A_phn)
            np.save(args.exp_dir + '/preds/word_pred.npy', valid_word_pred)
            np.save(args.exp_dir + '/preds/utt_pred.npy', A_utt)

    return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr

def valid_phn(audio_output, target):
    valid_token_pred = []
    valid_token_target = []
    audio_output = audio_output.squeeze(2)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            # only count valid tokens, not padded tokens (represented by negative values)
            if target[i, j] >= 0:
                valid_token_pred.append(audio_output[i, j])
                valid_token_target.append(target[i, j])
    valid_token_target = np.array(valid_token_target)
    valid_token_pred = np.array(valid_token_pred)

    valid_token_mse = np.mean((valid_token_target - valid_token_pred) ** 2)
    corr = np.corrcoef(valid_token_pred, valid_token_target)[0, 1]
    return valid_token_mse, corr

def valid_utt(audio_output, target):
    mse = []
    corr = []
    for i in range(5):
        cur_mse = np.mean(((audio_output[:, i] - target[:, i]) ** 2).numpy())
        cur_corr = np.corrcoef(audio_output[:, i], target[:, i])[0, 1]
        mse.append(cur_mse)
        corr.append(cur_corr)
    return mse, corr

def valid_word(audio_output, target):
    word_id = target[:, :, -1]
    target = target[:, :, 0:3]

    valid_token_pred = []
    valid_token_target = []

    # unique, counts = np.unique(np.array(target), return_counts=True)
    # print(dict(zip(unique, counts)))

    # for each utterance
    for i in range(target.shape[0]):
        prev_w_id = 0
        start_id = 0
        # for each token
        for j in range(target.shape[1]):
            cur_w_id = word_id[i, j].int()
            # if a new word
            if cur_w_id != prev_w_id:
                # average each phone belongs to the word
                valid_token_pred.append(np.mean(audio_output[i, start_id: j, :].numpy(), axis=0))
                valid_token_target.append(np.mean(target[i, start_id: j, :].numpy(), axis=0))
                # sanity check, if the range indeed contains a single word
                if len(torch.unique(target[i, start_id: j, 1])) != 1:
                    print(target[i, start_id: j, 0])
                # if end of the utterance
                if cur_w_id == -1:
                    break
                else:
                    prev_w_id = cur_w_id
                    start_id = j

    valid_token_pred = np.array(valid_token_pred)
    # this rounding is to solve the precision issue in the label
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list, corr_list = [], []
    # for each (accuracy, stress, total) word score
    for i in range(3):
        valid_token_mse = np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2)
        corr = np.corrcoef(valid_token_pred[:, i], valid_token_target[:, i])[0, 1]
        mse_list.append(valid_token_mse)
        corr_list.append(corr)
    return mse_list, corr_list, valid_token_pred, valid_token_target


class GoPDataset(Dataset):
    def __init__(self, set, am='librispeech'):
        # normalize the input to 0 mean and unit std.
        if am=='librispeech':
            dir='seq_data_librispeech'
            norm_mean, norm_std = 3.203, 4.045
        elif am=='paiia':
            dir='seq_data_paiia'
            norm_mean, norm_std = -0.652, 9.737
        elif am=='paiib':
            dir='seq_data_paiib'
            norm_mean, norm_std = -0.516, 9.247
        else:
            raise ValueError('Acoustic Model Unrecognized.')

        if set == 'train':
            self.feat = torch.tensor(np.load('../data/'+dir+'/tr_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/'+dir+'/tr_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/'+dir+'/tr_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/'+dir+'/tr_label_word.npy'), dtype=torch.float)
        elif set == 'test':
            self.feat = torch.tensor(np.load('../data/'+dir+'/te_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('../data/'+dir+'/te_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('../data/'+dir+'/te_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('../data/'+dir+'/te_label_word.npy'), dtype=torch.float)

        # normalize the GOP feature using the training set mean and std (only count the valid token features, exclude the padded tokens).
        self.feat = self.norm_valid(self.feat, norm_mean, norm_std)

        # normalize the utt_label to 0-2 (same with phn score range)
        self.utt_label = self.utt_label / 5
        # the last dim is word_id, so not normalizing
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5
        self.phn_label[:, :, 1] = self.phn_label[:, :, 1]

    # only normalize valid tokens, not padded token
    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        # feat, phn_label, phn_id, utt_label, word_label
        return self.feat[idx, :], self.phn_label[idx, :, 1], self.phn_label[idx, :, 0], self.utt_label[idx, :], self.word_label[idx, :]

args = parser.parse_args()

am = args.am
print('now train with {:s} acoustic models'.format(am))
feat_dim = {'librispeech':84, 'paiia':86, 'paiib': 88}
input_dim=feat_dim[am]

# nowa is the best models used in this work
if args.model == 'gopt':
    print('now train a GOPT models')
    audio_mdl = GOPT(embed_dim=args.embed_dim, num_heads=args.goptheads, depth=args.goptdepth, input_dim=input_dim)
# for ablation study only
elif args.model == 'gopt_nophn':
    print('now train a GOPT models without canonical phone embedding')
    audio_mdl = GOPTNoPhn(embed_dim=args.embed_dim, num_heads=args.goptheads, depth=args.goptdepth, input_dim=input_dim)
elif args.model == 'lstm':
    print('now train a baseline LSTM model')
    audio_mdl = BaselineLSTM(embed_dim=args.embed_dim, depth=args.goptdepth, input_dim=input_dim)

tr_dataset = GoPDataset('train', am=am)
tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
te_dataset = GoPDataset('test', am=am)
te_dataloader = DataLoader(te_dataset, batch_size=2500, shuffle=False)

train(audio_mdl, tr_dataloader, te_dataloader, args)