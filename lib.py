from fastai.conv_learner import *
from fastai.dataset import *
from pathlib import Path
from glob import glob
import tables as tb
from tqdm import tqdm
from tqdm import tqdm_notebook
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, 'code')
sys.path.insert(0, 'deeplab/pytorch-deeplab-resnet')
from v17 import *
import deeplab_resnet
from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from docopt import docopt
from functools import partial

BASE_DIR = "data/train"
BASE_TEST_DIR = "data/test"
WORKING_DIR = "data/working"
IMAGE_DIR = "data/working/images/{}".format('v5')

# ---------------------------------------------------------
# Image list, Image container and mask container
FMT_VALTRAIN_IM_FOLDER = IMAGE_DIR + "/trn_full_rgb_640/"
FMT_VALTEST_IM_FOLDER = IMAGE_DIR + "/test_full_rgb_640/"
FMT_TEST_IM_STORE = IMAGE_DIR + "/pred_full_rgb_640/"

FMT_VALTRAIN_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
FMT_VALTEST_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
FMT_VALTRAIN_IM_STORE = IMAGE_DIR + "/valtrain_{}_im.h5"
FMT_VALTEST_IM_STORE = IMAGE_DIR + "/valtest_{}_im.h5"
# FMT_VALTRAIN_MASK_STORE = IMAGE_DIR + "/valtrain_{}_mask.h5"
# FMT_VALTEST_MASK_STORE = IMAGE_DIR + "/valtest_{}_mask.h5"
FMT_VALTRAIN_MASK_STORE = IMAGE_DIR + "/valtrain_{}_mask.h5"
FMT_VALTEST_MASK_STORE = IMAGE_DIR + "/valtest_{}_mask.h5"
# FMT_VALTRAIN_MUL_STORE = IMAGE_DIR + "/valtrain_{}_mul.h5"
# FMT_VALTEST_MUL_STORE = IMAGE_DIR + "/valtest_{}_mul.h5"
# FMT_VALTRAIN_MUL_STORE = IMAGE_DIR + "/valtrain_{}_mul.h5"
# FMT_VALTEST_MUL_STORE = IMAGE_DIR + "/valtest_{}_mul.h5"

# FMT_TRAIN_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_train_ImageId.csv"
FMT_TEST_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
# FMT_TRAIN_IM_STORE = IMAGE_DIR + "/train_{}_im.h5"
# FMT_TRAIN_MASK_STORE = IMAGE_DIR + "/train_{}_mask.h5"
# FMT_TRAIN_MUL_STORE = IMAGE_DIR + "/train_{}_mul.h5"
# FMT_TEST_MUL_STORE = IMAGE_DIR + "/test_{}_mul.h5"
FMT_IMMEAN = IMAGE_DIR + "/{}_immean.h5"
# FMT_MULMEAN = IMAGE_DIR + "/{}_mulmean.h5"

# ---------------------------------------------------------
# Prediction & polygon result
FMT_TESTPRED_PATH = MODEL_DIR + "/{}_pred.h5"
FMT_VALTESTPRED_PATH = MODEL_DIR + "/{}_eval_pred.h5"
FMT_VALTESTPOLY_PATH = MODEL_DIR + "/{}_eval_poly.csv"
FMT_VALTESTTRUTH_PATH = MODEL_DIR + "/{}_eval_poly_truth.csv"
FMT_VALTESTPOLY_OVALL_PATH = MODEL_DIR + "/eval_poly.csv"
FMT_VALTESTTRUTH_OVALL_PATH = MODEL_DIR + "/eval_poly_truth.csv"
FMT_TESTPOLY_PATH = MODEL_DIR + "/{}_poly.csv"
FN_SOLUTION_CSV = "data/output/{}.csv".format(MODEL_NAME)


################################ models
(trn_x,trn_y), (val_x,val_y) = (None, None), (None, None)
class GlobalArraysDataset(BaseDataset):
    def __init__(self, is_trn, no_y, transform, num_slice=25, sz=256, rescale=False, is_test=False):
        self.is_trn = is_trn
        self.no_y = no_y
        # only val_x always exists, trn_x is missing when testing
        self.sz_i = val_x[0].shape[1]
        self.sz = sz
        self.num_slice = num_slice
        self.rescale = rescale
        self.is_test = is_test
        self.padding_sz = 59
        if is_test:
            self.sz_i = self.sz_i + 2 * self.padding_sz
        super().__init__(transform)

    def get_im(self, i, is_y):
        new_i = i if self.rescale else i//self.num_slice
        if is_y:
            im = trn_y[new_i] if self.is_trn else val_y[new_i]
        else:
            im = trn_x[new_i] if self.is_trn else val_x[new_i]

        if self.rescale:
            im = np.copy(im)
            if self.sz != self.sz_i:
                im = skimage.transform.resize(im, (self.sz, self.sz))
            return im
        else:
            if self.is_test:
                im = np.pad(im, ((self.padding_sz, self.padding_sz),
                    (self.padding_sz, self.padding_sz), (0, 0)),
                    'reflect')
            slice_pos = i % self.num_slice
            a = np.sqrt(self.num_slice)
            cut_j = slice_pos // a
            cut_i = slice_pos % a
            stride = (self.sz_i - self.sz) // (a - 1)
            cut_x = int(cut_j * stride)
            cut_y = int(cut_i * stride)
            return im[cut_x:cut_x + self.sz, cut_y:cut_y + self.sz]

    def get_x(self, i): return self.get_im(i, False)
    def get_y(self, i):
        if self.no_y:
            # just return a placeholder
            return self.get_im(i, False)[0]
        return self.get_im(i, True)
    def get_n(self):
        if self.rescale:
            return val_x.shape[0]
        else:
            return val_x.shape[0] * self.num_slice
    def get_sz(self):
        return self.sz  
    def get_c(self): return 1
    def denorm(self, arr):
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return np.clip(self.transform.denorm(np.rollaxis(arr,1,4)), 0, 1)

class UpsampleModel():
    def __init__(self, model, cut_base=8, name='upsample'):
        self.model,self.name = model,name
        self.cut_base = cut_base

    def get_layer_groups(self, precompute):
        c = children(self.model.module)
        return [c[:self.cut_base],
                c[self.cut_base:]]

#######################################   dataset
def get_rgb_mean_stat(area_id):
    prefix = area_id_to_prefix(area_id)

    with tb.open_file(FMT_IMMEAN.format(prefix), 'r') as f:
        im_mean = np.array(f.get_node('/immean'))[:3]
    mean = im_mean.mean(axis=(1, 2))
    std = imagenet_stats[1]
    return mean, std

def get_data(area_id, is_test, max_workers=3, debug=False, is_pred=False, X_only=False):
    prefix = area_id_to_prefix(area_id)
    if debug:
        slice_n = 9
    else:
        slice_n = None

    if is_pred:
        fn_train = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_im = FMT_TEST_IM_STORE
    else:
        fn_train = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix) if is_test\
                else FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        fn_im = FMT_VALTEST_MASK_STORE.format(prefix) if is_test\
                else FMT_VALTRAIN_MASK_STORE.format(prefix)
    df_train = pd.read_csv(fn_train)

    if not X_only:
        y = np.empty((slice_n if debug else df_train.shape[0], ORIGINAL_SIZE, ORIGINAL_SIZE, 1))
        with tb.open_file(fn_im, 'r') as f:
            for i, image_id in tqdm_notebook(enumerate(df_train.ImageId.tolist()[:slice_n]),\
                    total=df_train.shape[0], desc='gt'):
                fn = '/' + image_id
                y[i] = np.array(f.get_node(fn))[..., None]
    else:
        y = None

    # always get X
    if is_pred:
        fn_im = FMT_TEST_IM_STORE
    else:
        fn_im = FMT_VALTEST_IM_FOLDER if is_test else FMT_VALTRAIN_IM_FOLDER

    X = np.empty((slice_n if debug else df_train.shape[0], ORIGINAL_SIZE, ORIGINAL_SIZE, 3))
    if max_workers == 1:
        for i, image_id in tqdm_notebook(enumerate(df_train.ImageId.tolist()[:slice_n]),\
                total=df_train.shape[0], desc='ims'):
            X[i] = plt.imread(fn_im + image_id + '.png')[...,:3]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as e:
            gen = e.map(plt.imread, [fn_im + image_id + '.png'\
                    for image_id in df_train.ImageId.tolist()[:slice_n]])
            for i, im in enumerate(gen):
                X[i] = im[...,:3]

    X = X.astype('float')
    return X, y

def get_dataset(datapath, debug=False, is_eval=False, is_pred=False):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)

    X_only = is_pred
    val_x, val_y = get_data(area_id, is_test=True, debug=debug, is_pred=is_pred, X_only=X_only)
    if not X_only:
        val_y = np.broadcast_to(val_y, [val_y.shape[0], ORIGINAL_SIZE, ORIGINAL_SIZE, 3])
    if not is_pred and not is_eval:
        trn_x, trn_y = get_data(area_id, is_test=False, debug=debug, is_pred=is_pred, X_only=X_only)
        # training set is always with y
        trn_y = np.broadcast_to(trn_y, [trn_y.shape[0], ORIGINAL_SIZE, ORIGINAL_SIZE, 3])
    else:
        trn_x, trn_y = None, None
    return (trn_x,trn_y), (val_x,val_y)

def get_md_model(datapaths, data, bs, device_ids, num_workers, model_name='unet',
        num_slice=9, sz=256, no_y=False, rescale=False, **kwargs):
#     (trn_x, trn_y), (val_x, val_y) = trn, val
    global trn_x,trn_y, val_x,val_y

    aug_tfms = transforms_top_down
    for o in aug_tfms: o.tfm_y = TfmType.CLASS

    area_ids = [directory_name_to_area_id(datapath) for datapath in datapaths]
    stats = get_rgb_mean_stat(area_ids[0])
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
    #tfms = tfms_from_model(vgg16, sz // 32 * 32, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)

    (trn_x,trn_y), (val_x,val_y) = data
    trn, val = ((True, no_y), (False, no_y))
    datasets = ImageData.get_ds(GlobalArraysDataset, trn, val, tfms, num_slice=num_slice, sz=sz,
            rescale=rescale, **kwargs)
    md = ImageData('data', datasets, bs, num_workers=num_workers, classes=None)
    denorm = md.trn_ds.denorm

    if not Path(MODEL_DIR).exists():
        Path(MODEL_DIR).mkdir(parents=True)

    if model_name == 'deeplab':
        model = deeplab_resnet.Res_Deeplab(1) # change to 1
        cut_base = 1
    elif model_name == 'deeplab2':
        model = deeplab_resnet.Res_Deeplab(2)
        cut_base = 1
    elif model_name == 'unet':
        model = UNet16(pretrained='vgg')
        cut_base = 8
    elif model_name == 'linknet':
        model = LinkNet34(pretrained=True)
        cut_base = 8

    net = model.cuda()
    net = nn.DataParallel(net, device_ids)
    models = UpsampleModel(net, cut_base=cut_base)
    return md, models, denorm

def get_learn(md, model):
    learn=ConvLearner(md, model)
    learn.opt_fn=optim.Adam
    learn.crit=expanded_loss
    learn.metrics=[mask_acc, jaccard_coef_par]
    return learn

def learner_on_dataset(datapath, bs, device_ids, num_workers, model_name='unet', debug=False,
        data=None, num_slice=9, sz=256, is_eval=False, is_pred=False, rescale=False):
    if data is None:
        data = get_dataset(datapath, debug=debug, is_eval=is_eval, is_pred=is_pred)
    no_y = is_pred
    is_test = is_pred or is_eval
    md, model, denorm = get_md_model([datapath], data, bs, device_ids,
            num_workers, model_name=model_name, num_slice=num_slice, sz=sz,
            no_y=no_y, rescale=rescale, is_test=is_test)
    print('City finished loading:', datapath)
    learn = get_learn(md, model)
    return learn, denorm, data




################################### Loss
def expanded_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred[:,0], target) -\
            torch.log(jaccard_coef_loss(pred[:,0], target))
def jaccard_coef_loss(y_pred, y_true):
    epsilon = 1e-12
    y_pred_pos = torch.clamp(y_pred, 0, 1)
    intersection = torch.sum(y_true * y_pred_pos)
    sum_ = torch.sum(y_true + y_pred_pos)
    return (intersection + epsilon) / (sum_ - intersection + epsilon)
def jaccard_coef_par(pred, target):
    return jaccard_coef_loss(pred[:,0], target)
def fscore(pred, true, thresh=0.5):
    return 2 * torch.sum(true * pred + 1e-12) / torch.sum(true + pred + 1e-12)
def mask_acc(pred,targ): return accuracy_multi(pred[:,0], targ, 0.)

def sep_parallel(f, y_pred, y_true, num_workers=8, **kwargs):
    scores = np.empty(y_true.shape[0])
    if num_workers == 0:
        for i, (p, y) in enumerate(zip(y_pred, y_true)):
            scores[i] = f(p, y, **kwargs)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as e:
            res = e.map(partial(f, **kwargs), y_pred, y_true)
            for i, score in enumerate(res):
                scores[i] = score
    return scores



###################################   Plots
def plot_lr_loss(learn, save_name=None):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.tight_layout()
    ax[0].plot(learn.sched.iterations, learn.sched.losses)
    ax[0].set_xlabel('loss')
    ax[1].plot(learn.sched.iterations, learn.sched.lrs)
    ax[1].set_xlabel('lr')
    if save_name is not None:
        save_path = Path('data/figs')
        if not save_path.exists(): save_path.mkdir(parents=True)
        fig.savefig(str(save_path / Path(save_name)) + '.png')

def train_and_plot(learn, idx, fn, lrs, n_cycles, wds=[0.025/3, 0.025], use_wd_sched=False, **kwargs):
    learn.fit(lrs, n_cycles, wds=wds, **kwargs)
    save_name = fn + '_' + str(idx)
    learn.save(save_name)
    plot_lr_loss(learn, save_name)

def plot_ims(data, labels=None, figsize=3):
    # data and labels should be zips
    data = list(data)
    n_ims = len(data)
    cols = len(list(zip(*data)))
    if labels is not None:
        labels = list(labels)
    else:
        labels = np.zeros((n_ims, cols))
    fig, ax = plt.subplots(n_ims, cols, figsize=(figsize*cols, figsize*n_ims))
    for i, row in enumerate(ax):
        if len(ax.shape) == 1:
            row.imshow(data[0][i])
            row.set_xlabel(labels[0][i])
        else:
            for j, col in enumerate(row):
                col.imshow(data[i][j])
                col.set_xlabel(labels[i][j])
    fig.tight_layout()

def plot_worse_preds(x, y, preds, crit=None, scores=None, shift=0,
        n_ims=9, is_best=False, thresh=0.5, denorm=None, **kwargs):
    if scores is None:
        scores = sep_parallel(crit, preds, y, **kwargs, num_workers=0)
    lowest_iou_idx = np.argsort(scores)
    if is_best:
        lowest_iou_idx = np.flip(lowest_iou_idx, 0)
    lowest_iou_idx = lowest_iou_idx[shift: n_ims + shift]
    print(scores[lowest_iou_idx])
    labels = [lowest_iou_idx,
        ["gt"] * n_ims,
        scores[lowest_iou_idx],
        ["preds"] * n_ims]
    bp = bool_pred(preds, thresh=thresh)
    xl, yl, predsl, bpl = [], [], [], []
    for i in lowest_iou_idx:
        if denorm is not None:
            xl.append(learn.data.trn_ds.denorm(x[i]).squeeze())
        else:
            xl.append(x[i])
        yl.append(y[i])
        predsl.append(preds[i])
        bpl.append(bp[i])
    data = zip(xl, yl, predsl, bpl)
    labels = zip(*labels)
    plot_ims(data, labels=labels)
    return scores, lowest_iou_idx






































