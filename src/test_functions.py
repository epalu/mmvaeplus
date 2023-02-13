import os
import numpy as np
import glob
from fid.inception import InceptionV3
from fid.fid_score import get_activations
from fid.fid_score import calculate_frechet_distance
from sklearn.metrics import accuracy_score

def calculate_inception_features_for_gen_evaluation(inception_state_dict_path, device, dir_fid_base, datadir, dims=2048, batch_size=128):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=inception_state_dict_path)
    model = model.to(device)

    for moddality_num in range(5):
        moddality = 'm{}'.format(moddality_num)
        filename_act_real_calc = os.path.join(datadir, 'test','real_activations_{}.npy'.format(moddality))
        if not os.path.exists(filename_act_real_calc):
            files_real_calc = glob.glob(os.path.join(datadir,  'test', moddality, '*' + '.png'))
            act_real_calc = get_activations(files_real_calc, model, device, batch_size, dims, verbose=False)
            np.save(filename_act_real_calc, act_real_calc)

    for prefix  in ['random', 'm0', 'm1', 'm2', 'm3', 'm4']:
        dir_gen = os.path.join(dir_fid_base, prefix)
        if not os.path.exists(dir_gen):
            raise RuntimeError('Invalid path: %s' % dir_gen)
        for modality in ['m{}'.format(m) for m in range(5)]:
            files_gen = glob.glob(os.path.join(dir_gen, modality, '*' + '.png'))
            filename_act = os.path.join(dir_gen,
                                           modality + '_activations.npy')
            act_rand_gen = get_activations(files_gen, model, device, batch_size, dims, verbose=False)
            np.save(filename_act, act_rand_gen)



def load_inception_activations(flags, modality=None, num_modalities=2, conditionals=None):
    if modality is None:
        filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_img_activations.npy');
        filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_img_activations.npy')
        filename_conditional = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'conditional_img_activations.npy')
        feats_real = np.load(filename_real);
        feats_random = np.load(filename_random);
        feats_cond = np.load(filename_conditional);
        feats = [feats_real, feats_random, feats_cond];
    else:
        filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_' + modality + '_activations.npy');
        filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_sampling_' + modality + '_activations.npy')
        feats_real = np.load(filename_real);
        feats_random = np.load(filename_random);

        #if num_modalities == 2:
            #filename_cond_gen = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'cond_gen_' + modality + '_activations.npy')
            #feats_cond_gen = np.load(filename_cond_gen);
            #feats = [feats_real, feats_random, feats_cond_gen];
        #elif num_modalities > 2:
            #if conditionals is None:
                #raise RuntimeError('conditionals are needed for num(M) > 2...')
        feats_cond_1a2m = dict()
        for k, key in enumerate(conditionals[0].keys()):
            filename_cond_1a2m = os.path.join(conditionals[0][key], key + '_' + modality + '_activations.npy')
            feats_cond_key = np.load(filename_cond_1a2m);
            feats_cond_1a2m[key] = feats_cond_key
        feats = [feats_real, feats_random, feats_cond_1a2m] #, feats_cond_2a1m, feats_cond_dyn_prior_2a1m];
        #else:
            #print('combinations of feature names and number of modalities is not correct');
    return feats;

def calculate_fid(feats_real, feats_gen):
    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid;


def calculate_fid_dict(feats_real, dict_feats_gen):
    dict_fid = dict();
    for k, key in enumerate(dict_feats_gen.keys()):
        feats_gen = dict_feats_gen[key];
        dict_fid[key] = calculate_fid(feats_real, feats_gen);
    return dict_fid;


def get_clf_activations(flags, data, model):
    model.eval();
    act = model.get_activations(data);
    act = act.cpu().data.numpy().reshape(flags.batch_size, -1)
    return act;

def classify_latent_representations(clf_lr, data, labels):
    print('Clslfying lrs')
    gt = np.argmax(labels, axis=1).astype(int)
    accuracies = dict()
    for k, data_k in enumerate(data):         
        data_rep_u, data_rep_w, data_rep_z = data_k

        clf_key_u = 'm' + str(k) + '_'+'u'
        clf_lr_rep_u = clf_lr[clf_key_u];
        y_pred_rep_u = clf_lr_rep_u.predict(data_rep_u);
        accuracy_rep_u = accuracy_score(gt, y_pred_rep_u.ravel());
        accuracies[clf_key_u] = accuracy_rep_u;

        clf_key_z = 'm' + str(k) + '_' + 'z'
        clf_lr_rep_z = clf_lr[clf_key_z];
        y_pred_rep_z = clf_lr_rep_z.predict(data_rep_z);
        accuracy_rep_z = accuracy_score(gt, y_pred_rep_z.ravel());
        accuracies[clf_key_z] = accuracy_rep_z;

        clf_key_w = 'm' + str(k) + '_' + 'w'
        clf_lr_rep_w = clf_lr[clf_key_w];
        y_pred_rep_w = clf_lr_rep_w.predict(data_rep_w);
        accuracy_rep_w = accuracy_score(gt, y_pred_rep_w.ravel());
        accuracies[clf_key_w] = accuracy_rep_w;
    return accuracies;








