import argparse
import datetime
import sys
import json
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
from torch import optim
import models
import objectives
from utils import Logger, Timer, save_model_light, save_vars, unpack_data_polymnist, get_10_polyMNIST_samples
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
from test_functions import calculate_inception_features_for_gen_evaluation, calculate_fid, classify_latent_representations
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LogisticRegression
from models.clf_polyMNIST import ClfImg
from utils import NonLinearLatent_Classifier

parser = argparse.ArgumentParser(description='Multi-Modal VAEs')
parser.add_argument('--experiment', type=str, default='', metavar='E',
                    help='experiment name')
parser.add_argument('--obj', type=str, default='elbo', metavar='O',
                    choices=[ 'elbo',  'dreg'],
                    help='objective to use (default: elbo)')
parser.add_argument('--K', type=int, default=1, metavar='K',
                    help='number of particles to use for iwae/dreg (default: 10)')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='batch size for data (default: 256)')
parser.add_argument('--epochs', type=int, default=150, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--latent-dim-w', type=int, default=32)
parser.add_argument('--latent-dim-z', type=int, default=32)
parser.add_argument('--print-freq', type=int, default=1, metavar='f',
                    help='frequency with which to print stats (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--variant', type=str, default='mmvaeplus',
                    choices=['mmvaeplus', 'mmvaefactorized'])
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--tmpdir', type=str, default='/data')
parser.add_argument('--outputdir', type=str, default='./outputs')
parser.add_argument('--inception_module_path', type=str, default='../data/pt_inception-2015-12-05-6726825d.pth')
parser.add_argument('--pretrained_clfs_dir', type=str, default='../data/PolyMNIST/trained_clfs_polyMNIST')

# args
args = parser.parse_args()
flags_clf_lr = {'latdimu': args.latent_dim_z,
                'latdimw': args.latent_dim_w}
# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load args from disk if pretrained model path is given
#pretrained_path = ""
#if args.pre_trained:
#    pretrained_path = args.pre_trained
#    args = torch.load(args.pre_trained + '/args.rar')

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

args.model = "polymnist_5modalities"
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)

if args.obj == 'elbo': 
    args.K = 1

#if pretrained_path:
#    print('Loading model {} from {}'.format(model.modelName, pretrained_path))
#    model.load_state_dict(torch.load(pretrained_path + '/model.rar'))
#    model._pu_params = model._pu_params

if not args.experiment:
    args.experiment = model.modelName

# set up run path
runId = str(args.latent_dim_w) + '_' + str(args.latent_dim_z) + '_' + str(args.beta) + '_' + str(args.seed) + \
        '_' + datetime.datetime.now().isoformat()
experiment_dir = Path(os.path.join(args.outputdir, 'saved_models', args.experiment))
experiment_dir.mkdir(parents=True, exist_ok=True)
runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Saving models and lossed at:', runPath)
print('RunID:', runId)

NUM_VAES = len(model.vaes)
# Define path where to save images for FID score calculation. By default this happens in tmpdir
fid_path = os.path.join(args.tmpdir, 'fids_' + (runPath.rsplit('/')[-1]))
datadir = os.path.join(args.tmpdir, "PolyMNIST")

# Save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

# TensorBoard logging
tensorboard_log_dir = args.outputdir + "/runs/" + args.experiment + "/" + runId
writer = SummaryWriter(log_dir=tensorboard_log_dir)

objectives = objectives

# Preparation for training
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, amsgrad=True)
# Data loaders
train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)

# Define objective
objective = getattr(objectives, 'm_'+ args.obj)

# Cuda stuff
needs_conversion = not args.cuda
conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}

# Load pretrained classifiers for each modality
clfs = [ClfImg() for idx, modal in enumerate(model.vaes)]
for idx, vae in enumerate(model.vaes):
    clfs[idx].load_state_dict(torch.load(args.pretrained_clfs_dir+"/pretrained_img_to_digit_clf_m"+str(idx), **conversion_kwargs), strict=False)
    clfs[idx].eval()
    if args.cuda:
        clfs[idx].cuda()

def train(epoch, agg):
    model.train()
    b_loss = 0
    for i, dataT in enumerate(train_loader):
        data, labels_batch = unpack_data_polymnist(dataT, device=device)
        optimizer.zero_grad()
        loss = -objective(model, data, K=args.K, test=False)
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if args.print_freq > 0 and i % args.print_freq == 0:
            print("iteration {:04d}: loss: {:6.3f}".format(i, loss.item() / args.batch_size))
    epoch_loss = b_loss / len(train_loader.dataset)
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    agg['train_loss'].append(epoch_loss)
    print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, agg['train_loss'][-1]))

def test(epoch, agg):
    model.eval()
    b_loss = 0
    with torch.no_grad():
        test_selected_samples = get_10_polyMNIST_samples(test_loader.dataset,
                                                     num_testing_images=test_loader.dataset.__len__(), device=device)
        for i, dataT in enumerate(test_loader):
            data, _ = unpack_data_polymnist(dataT, device=device)
            loss = -objective(model, data, K=args.K, test=True)
            b_loss += loss.item()
            if i == 0:
                cg_imgs = model.cross_generate(test_selected_samples)
                for i in range(NUM_VAES):
                    for j in range(NUM_VAES):
                        writer.add_image(tag='Cross_Generation/m{}/m{}'.format(i, j), img_tensor=cg_imgs[i][j],
                                         global_step=epoch)
    epoch_loss = b_loss / len(test_loader.dataset)
    writer.add_scalar("Loss/test", epoch_loss, epoch)
    agg['test_loss'].append(epoch_loss)
    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))

def cross_coherence():
    "Calculates cross-modal coherence"
    model.eval()
    corrs = [[0 for idx, modal in enumerate(model.vaes)] for idx, modal in enumerate(model.vaes)]
    total = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data, targets = unpack_data_polymnist(dataT, device)  # needs to be sent to device
            total += targets.size(0)
            _, px_us, _ = model.reconstruct_and_cross_reconstruct_forw(data)
            for idx_srt, srt_mod in enumerate(model.vaes):
                for idx_trg, trg_mod in enumerate(model.vaes):
                    clfs_results = torch.argmax(clfs[idx_trg](px_us[idx_srt][idx_trg].mean.squeeze(0)), dim=-1)
                    corrs[idx_srt][idx_trg] += (clfs_results == targets).sum().item()
        for idx_trgt, vae in enumerate(model.vaes):
            for idx_strt, _ in enumerate(model.vaes):
                corrs[idx_strt][idx_trgt] = corrs[idx_strt][idx_trgt] / total
        means_target = [0 for idx, modal in enumerate(model.vaes)]
        for idx_target, _ in enumerate(model.vaes):
            means_target[idx_target] = mean(
                [corrs[idx_start][idx_target] for idx_start, _ in enumerate(model.vaes) if idx_start != idx_target])
    return corrs, means_target, mean(means_target)

def unconditional_coherence_and_lr(clf_lr):
    "Evaluates unconditinal coherence and latent classification accuracy with linear digit classifiers"
    model.eval()
    correct = 0
    total = 0
    lr_acc_m0_u, lr_acc_m1_u, lr_acc_m2_u, lr_acc_m3_u, lr_acc_m4_u = [], [], [], [], []
    lr_acc_m0_w, lr_acc_m1_w, lr_acc_m2_w, lr_acc_m3_w, lr_acc_m4_w = [], [], [], [], []
    lr_acc_m0_z, lr_acc_m1_z, lr_acc_m2_z, lr_acc_m3_z, lr_acc_m4_z = [], [], [], [], []
    accuracies_lr = {'u':{},
                     'z':{},
                     'w':{}}
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            # Unconditional coherence
            data, targets = unpack_data_polymnist(dataT, device)
            b_size = data[0].size(0)
            labels_batch = nn.functional.one_hot(targets, num_classes=10).float()
            labels = labels_batch.cpu().data.numpy().reshape(b_size, 10)
            uncond_gens = model.generate_for_calculating_unconditional_coherence(b_size)
            uncond_gens = [elem.to(device) for elem in uncond_gens]
            clfs_resultss = []
            for idx_trg, trg_mod in enumerate(model.vaes):
                clfs_results = torch.argmax(clfs[idx_trg](uncond_gens[idx_trg]), dim=-1)
                if idx_trg == 0:
                    total += b_size
                clfs_resultss.append(clfs_results)
            clfs_resultss_tensor = torch.stack(clfs_resultss, dim=-1)
            for dim in range(clfs_resultss_tensor.size(0)):
                if torch.unique(clfs_resultss_tensor[dim, :]).size(0) == 1:
                    correct += 1
            # Evaluate learnt representations (linear digit classifiers)
            if clf_lr is not None:
                latent_reps = []
                for v, vae in enumerate(model.vaes):
                    with torch.no_grad():
                        qu_x_params = vae.enc(data[v])
                        us_v = vae.qu_x(*qu_x_params).rsample()
                    ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
                    latent_reps.append([us_v.cpu().data.numpy(), ws_v.cpu().data.numpy(), zs_v.cpu().data.numpy()])
                accuracies = classify_latent_representations(clf_lr, latent_reps, labels)

                lr_acc_m0_u.append(np.mean(accuracies['m0_u']))
                lr_acc_m1_u.append(np.mean(accuracies['m1_u']))
                lr_acc_m2_u.append(np.mean(accuracies['m2_u']))
                lr_acc_m3_u.append(np.mean(accuracies['m3_u']))
                lr_acc_m4_u.append(np.mean(accuracies['m4_u']))

                lr_acc_m0_w.append(np.mean(accuracies['m0_w']))
                lr_acc_m1_w.append(np.mean(accuracies['m1_w']))
                lr_acc_m2_w.append(np.mean(accuracies['m2_w']))
                lr_acc_m3_w.append(np.mean(accuracies['m3_w']))
                lr_acc_m4_w.append(np.mean(accuracies['m4_w']))

                lr_acc_m0_z.append(np.mean(accuracies['m0_z']))
                lr_acc_m1_z.append(np.mean(accuracies['m1_z']))
                lr_acc_m2_z.append(np.mean(accuracies['m2_z']))
                lr_acc_m3_z.append(np.mean(accuracies['m3_z']))
                lr_acc_m4_z.append(np.mean(accuracies['m4_z']))

        uncond_coherence = correct / total

        accuracies_lr['u']['m0'] = mean(lr_acc_m0_u)
        accuracies_lr['u']['m1'] = mean(lr_acc_m1_u)
        accuracies_lr['u']['m2'] = mean(lr_acc_m2_u)
        accuracies_lr['u']['m3'] = mean(lr_acc_m3_u)
        accuracies_lr['u']['m4'] = mean(lr_acc_m4_u)

        accuracies_lr['w']['m0'] = mean(lr_acc_m0_w)
        accuracies_lr['w']['m1'] = mean(lr_acc_m1_w)
        accuracies_lr['w']['m2'] = mean(lr_acc_m2_w)
        accuracies_lr['w']['m3'] = mean(lr_acc_m3_w)
        accuracies_lr['w']['m4'] = mean(lr_acc_m4_w)

        accuracies_lr['z']['m0'] = mean(lr_acc_m0_z)
        accuracies_lr['z']['m1'] = mean(lr_acc_m1_z)
        accuracies_lr['z']['m2'] = mean(lr_acc_m2_z)
        accuracies_lr['z']['m3'] = mean(lr_acc_m3_z)
        accuracies_lr['z']['m4'] = mean(lr_acc_m4_z)

        accuracies_lr['u']['avg'] = mean([accuracies_lr['u']['m{}'.format(n)] for n in range(NUM_VAES)])
        accuracies_lr['w']['avg'] = mean([accuracies_lr['w']['m{}'.format(n)] for n in range(NUM_VAES)])
        accuracies_lr['z']['avg'] = mean([accuracies_lr['z']['m{}'.format(n)] for n in range(NUM_VAES)])

    return uncond_coherence, accuracies_lr

def classify_latents_nl(mod, epoch):
    model.eval()
    classifier_w = NonLinearLatent_Classifier(args.latent_dim_w).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_w = optim.Adam(classifier_w.parameters(), lr=0.001)
    # Train classifier
    running_loss_w = 0.0
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, dataT in enumerate(train_loader):
        data, labels = unpack_data_polymnist(dataT, device=device)
        data_batch = data[mod]
        with torch.no_grad():
            qu_x_params = model.vaes[mod].enc(data_batch)
            us = model.vaes[mod].qu_x(*qu_x_params).rsample()
            ws, _ = torch.split(us, [args.latent_dim_w, args.latent_dim_z], dim=-1)
        optimizer_w.zero_grad()
        outputs_w = classifier_w(ws)
        loss_w = criterion(outputs_w, labels)
        loss_w.backward()
        optimizer_w.step()
        # print statistics
        running_loss_w += loss_w.item()
    print('Finished Training, calculating test loss...')
    classifier_w.eval()
    total = 0
    correct_w = 0
    with torch.no_grad():
        for i, dataT in enumerate(test_loader):
            data, labels = unpack_data_polymnist(dataT, device=device)
            data_batch = data[mod]
            with torch.no_grad():
                qu_x_params = model.vaes[mod].enc(data_batch)
                us = model.vaes[mod].qu_x(*qu_x_params).rsample()
                ws, _ = torch.split(us, [args.latent_dim_w, args.latent_dim_z], dim=-1)
            outputs_w = classifier_w(ws)
            _, predicted_w = torch.max(outputs_w.data, 1)
            total += labels.size(0)
            correct_w += (predicted_w == labels).sum().item()
    writer.add_scalar("Latent_classification_accuracy_nl_w/m{}".format(mod), (correct_w / total), global_step=epoch)

def calculate_fid_routine(datadir, fid_path, num_fid_samples, epoch):
    "Calculates FID scores"
    total_cond = 0
    # Create new directories for conditional FIDs
    for j in [0, 1, 2, 3, 4]:
        if os.path.exists(os.path.join(fid_path, 'random', 'm{}'.format(j))):
            shutil.rmtree(os.path.join(fid_path, 'random', 'm{}'.format(j)))
            os.makedirs(os.path.join(fid_path, 'random', 'm{}'.format(j)))
        else:
            os.makedirs(os.path.join(fid_path, 'random', 'm{}'.format(j)))
        for i in [0, 1, 2, 3, 4]:
            if os.path.exists(os.path.join(fid_path, 'm{}'.format(j), 'm{}'.format(i))):
                shutil.rmtree(os.path.join(fid_path, 'm{}'.format(j), 'm{}'.format(i)))
                os.makedirs(os.path.join(fid_path, 'm{}'.format(j), 'm{}'.format(i)))
            else:
                os.makedirs(os.path.join(fid_path, 'm{}'.format(j), 'm{}'.format(i)))
    with torch.no_grad():
        # Generate unconditional fid samples
        for tranche in range(num_fid_samples // 100):
            model.generate_for_fid(fid_path, 100, tranche)
        # Generate conditional fid samples
        for i, dataT in enumerate(test_loader):
            data, _ = unpack_data_polymnist(dataT, device=device)
            if total_cond < num_fid_samples:
                model.reconstruct_for_fid(data, fid_path, i)
                total_cond += data[0].size(0)
        calculate_inception_features_for_gen_evaluation(args.inception_module_path, device, fid_path, datadir)
        # FID calculation
        fid_randm_list = []
        fid_condgen_list = []
        for modality_target in ['m{}'.format(m) for m in range(5)]:
            file_activations_real = os.path.join(args.tmpdir, 'PolyMNIST', 'test',
                                                 'real_activations_{}.npy'.format(modality_target))
            feats_real = np.load(file_activations_real)
            file_activations_randgen = os.path.join(fid_path, 'random',
                                                    modality_target + '_activations.npy')
            feats_randgen = np.load(file_activations_randgen)
            fid_randval = calculate_fid(feats_real, feats_randgen)
            writer.add_scalar("FID/{}/{}".format('random', modality_target), fid_randval, epoch)
            fid_randm_list.append(fid_randval)
            fid_condgen_target_list = []
            for modality_source in ['m{}'.format(m) for m in range(5)]:
                file_activations_gen = os.path.join(fid_path, modality_source,
                                                    modality_target + '_activations.npy')
                feats_gen = np.load(file_activations_gen)
                fid_val = calculate_fid(feats_real, feats_gen)
                writer.add_scalar("FID/{}/{}".format(modality_source, modality_target), fid_val, epoch)
                fid_condgen_target_list.append(fid_val)
            fid_condgen_list.append(mean(fid_condgen_target_list))
        mean_fid_condgen = mean(fid_condgen_list)
        mean_fid_randm = mean(fid_randm_list)
        writer.add_scalar("FID/random_overallavg", mean_fid_randm, epoch)
        writer.add_scalar("FID/condionalgeneration_overallavg", mean_fid_condgen, epoch)
    if os.path.exists(fid_path):
        shutil.rmtree(fid_path)
        os.makedirs(fid_path)


def train_clf_lr(dl):
    latent_rep = {'m0': {'u': [], 'z': [], 'w': []},
                  'm1': {'u': [], 'z': [], 'w': []},
                  'm2': {'u': [], 'z': [], 'w': []},
                  'm3': {'u': [], 'z': [], 'w': []},
                  'm4': {'u': [], 'z': [], 'w': []}}
    labels_all = []
    for i, dataT_lr in enumerate(dl):
        data, labels_batch = unpack_data_polymnist(dataT_lr, device=device)
        b_size = data[0].size(0)
        labels_batch = nn.functional.one_hot(labels_batch, num_classes=10).float()
        labels = labels_batch.cpu().data.numpy().reshape(b_size, 10);
        labels_all.append(labels)
        for v, vae in enumerate(model.vaes):
            with torch.no_grad():
                qu_x_params = vae.enc(data[v])
                us_v = vae.qu_x(*qu_x_params).rsample()
            ws_v, zs_v = torch.split(us_v, [args.latent_dim_w, args.latent_dim_z], dim=-1)
            latent_rep['m{}'.format(v)]['u'].append(us_v.cpu().data.numpy())
            latent_rep['m{}'.format(v)]['z'].append(zs_v.cpu().data.numpy())
            latent_rep['m{}'.format(v)]['w'].append(ws_v.cpu().data.numpy())
    labels_all = np.concatenate(labels_all, axis=0)
    gt = np.argmax(labels_all, axis=1).astype(int)
    clf_lr = dict();
    for v, vae in enumerate(model.vaes):
        latent_rep_u = np.concatenate(latent_rep['m{}'.format(v)]['u'], axis=0)
        latent_rep_w = np.concatenate(latent_rep['m{}'.format(v)]['w'], axis=0)
        latent_rep_z = np.concatenate(latent_rep['m{}'.format(v)]['z'], axis=0)
        clf_lr_rep_u = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
        clf_lr_rep_z = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
        clf_lr_rep_w = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000)
        clf_lr_rep_u.fit(latent_rep_u, gt.ravel())
        clf_lr['m' + str(v) + '_' + 'u'] = clf_lr_rep_u
        clf_lr_rep_w.fit(latent_rep_w, gt.ravel())
        clf_lr['m' + str(v) + '_' + 'w'] = clf_lr_rep_w
        clf_lr_rep_z.fit(latent_rep_z, gt.ravel())
        clf_lr['m' + str(v) + '_' + 'z'] = clf_lr_rep_z
    return clf_lr

if __name__ == '__main__':
    with Timer('MM-VAE') as t:
        agg = defaultdict(list)
        for epoch in range(1, args.epochs + 1):
            train(epoch, agg)
            if epoch % 25 == 0:
                test(epoch, agg)
                clf_lr = train_clf_lr(train_loader)
                save_model_light(model, runPath + '/model_' + str(epoch) + '.rar')
                save_vars(agg, runPath + '/losses_' + str(epoch) + '.rar')
                gen_samples = model.generate()
                for j in range(NUM_VAES):
                    writer.add_image(tag='Generation_m{}'.format(j), img_tensor=gen_samples[j],
                                     global_step=epoch)
                cors, means_tgt, mt = cross_coherence()
                writer.add_scalar("Conditional_coherence_overallavg", mt, global_step=epoch)
                for i in range(NUM_VAES):
                    writer.add_scalar("Conditional_coherence_avg_target_m{}".format(i), means_tgt[i], global_step=epoch)
                    for j in range(NUM_VAES):
                        writer.add_scalar("Conditional_coherence_m{}xm{}".format(i, j), cors[i][j], global_step=epoch)
                uncond_coher, accuracies_lr = unconditional_coherence_and_lr(clf_lr)
                writer.add_scalar("Unconditional_coherence", uncond_coher, global_step=epoch)
                for key_out in accuracies_lr.keys():
                    for key_in in accuracies_lr[key_out]:
                        writer.add_scalar("Latent_classification_accuracy_linear_{}/{}".format(key_out, key_in), accuracies_lr[key_out][key_in], global_step=epoch)
                for i_clf_nl in range(NUM_VAES):
                    classify_latents_nl(i_clf_nl, epoch)
                calculate_fid_routine(datadir, fid_path, 10000, epoch)
        writer.flush()
        writer.close()
