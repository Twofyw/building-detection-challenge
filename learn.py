from fastai.conv_learner import *
from fastai.dataset import *
from pathlib import Path
from glob import glob
import tables as tb
import tqdm
import multiprocessing as mp
import sys
sys.path.insert(0, 'code')
from models import *
from v13_deeplab import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'deeplab/pytorch-deeplab-resnet')
import deeplab_resnet
from docopt import docopt
from functools import partial

MODEL_NAME = 'v13'
ORIGINAL_SIZE = 650
sz = 256
num_slice = 9
STRIDE_SZ = 197
PATH = 'data/'

#_num_gpus, _gpu_start, _num_workers, _device_ids, _bs = None, None, None, None, None
#_debug = None
#learn, denorm = None, None
#(trn_x,trn_y), (val_x,val_y) = (None, None), (None, None)
#last_datapath = None

BASE_DIR = "data/train"
BASE_TEST_DIR = "data/test"
WORKING_DIR = "data/working"

# Restore later
IMAGE_DIR = "data/working/images/{}".format('v12')
# IMAGE_DIR = "data/working/images/{}".format('v5')
V5_IMAGE_DIR = "data/working/images/{}".format('v5')

# ---------------------------------------------------------
# Parameters
MIN_POLYGON_AREA = 30  # 30

# ---------------------------------------------------------
# Input files
FMT_TRAIN_SUMMARY_PATH = str(
            Path(BASE_DIR) /
                Path("{prefix:s}_Train/") /
                    Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))
FMT_TRAIN_RGB_IMAGE_PATH = str(
            Path(BASE_DIR) /
                Path("{prefix:s}_Train/") /
                    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TEST_RGB_IMAGE_PATH = str(
            Path(BASE_TEST_DIR) /
                Path("{prefix:s}_Test/") /
                    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TRAIN_MSPEC_IMAGE_PATH = str(
            Path(BASE_DIR) /
                Path("{prefix:s}_Train/") /
                    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
FMT_TEST_MSPEC_IMAGE_PATH = str(
            Path(BASE_TEST_DIR) /
                Path("{prefix:s}_Test/") /
                    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))

# ---------------------------------------------------------
# Preprocessing result
FMT_RGB_BANDCUT_TH_PATH = IMAGE_DIR + "/rgb_bandcut.csv"
FMT_MUL_BANDCUT_TH_PATH = IMAGE_DIR + "/mul_bandcut.csv"

# ---------------------------------------------------------
# Image list, Image container and mask container
FMT_VALTRAIN_IM_FOLDER = V5_IMAGE_DIR + "/trn_full_rgb/"
FMT_VALTEST_IM_FOLDER = V5_IMAGE_DIR + "/test_full_rgb/"

FMT_VALTRAIN_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
FMT_VALTEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
FMT_VALTRAIN_IM_STORE = IMAGE_DIR + "/valtrain_{}_im.h5"
FMT_VALTEST_IM_STORE = IMAGE_DIR + "/valtest_{}_im.h5"
# FMT_VALTRAIN_MASK_STORE = IMAGE_DIR + "/valtrain_{}_mask.h5"
# FMT_VALTEST_MASK_STORE = IMAGE_DIR + "/valtest_{}_mask.h5"
FMT_VALTRAIN_MASK_STORE = V5_IMAGE_DIR + "/valtrain_{}_mask.h5"
FMT_VALTEST_MASK_STORE = V5_IMAGE_DIR + "/valtest_{}_mask.h5"
# FMT_VALTRAIN_MUL_STORE = IMAGE_DIR + "/valtrain_{}_mul.h5"
# FMT_VALTEST_MUL_STORE = IMAGE_DIR + "/valtest_{}_mul.h5"
FMT_VALTRAIN_MUL_STORE = V5_IMAGE_DIR + "/valtrain_{}_mul.h5"
FMT_VALTEST_MUL_STORE = V5_IMAGE_DIR + "/valtest_{}_mul.h5"

FMT_TRAIN_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_train_ImageId.csv"
FMT_TEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
FMT_TRAIN_IM_STORE = IMAGE_DIR + "/train_{}_im.h5"
FMT_TEST_IM_STORE = IMAGE_DIR + "/test_{}_im.h5"
FMT_TRAIN_MASK_STORE = IMAGE_DIR + "/train_{}_mask.h5"
FMT_TRAIN_MUL_STORE = IMAGE_DIR + "/train_{}_mul.h5"
FMT_TEST_MUL_STORE = IMAGE_DIR + "/test_{}_mul.h5"
FMT_IMMEAN = V5_IMAGE_DIR + "/{}_immean.h5"
FMT_MULMEAN = IMAGE_DIR + "/{}_mulmean.h5"

# ---------------------------------------------------------
# Model files
MODEL_DIR = "data/working/models/{}".format(MODEL_NAME)
FMT_VALMODEL_PATH = MODEL_DIR + "/{}_val_weights.h5"
FMT_FULLMODEL_PATH = MODEL_DIR + "/{}_full_weights.h5"
FMT_VALMODEL_HIST = MODEL_DIR + "/{}_val_hist.csv"
FMT_VALMODEL_EVALHIST = MODEL_DIR + "/{}_val_evalhist.csv"
FMT_VALMODEL_EVALTHHIST = MODEL_DIR + "/{}_val_evalhist_th.csv"

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

# ---------------------------------------------------------
# Model related files (others)
FMT_VALMODEL_LAST_PATH = MODEL_DIR + "/{}_val_weights_last.h5"
FMT_FULLMODEL_LAST_PATH = MODEL_DIR + "/{}_full_weights_last.h5"

datapaths = ['data/train/AOI_3_Paris_Train', 'data/train/AOI_2_Vegas_Train', 'data/train/AOI_4_Shanghai_Train', 'data/train/AOI_5_Khartoum_Train']
################################################################

def get_data(area_id, is_test, max_workers=3, debug=False):
    prefix = area_id_to_prefix(area_id)
    fn_train = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix) if is_test\
	else FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
    df_train = pd.read_csv(fn_train)
    
    fn_im = FMT_VALTEST_MASK_STORE.format(prefix) if is_test\
	else FMT_VALTRAIN_MASK_STORE.format(prefix)
    if debug:
        slice_n = 50
    else:
        slice_n = None
    y_val = np.empty((slice_n if debug else df_train.shape[0], ORIGINAL_SIZE, ORIGINAL_SIZE, 1))
    with tb.open_file(fn_im, 'r') as f:                                         
        for i, image_id in tqdm.tqdm_notebook(enumerate(df_train.ImageId.tolist()[:slice_n]),\
		total=df_train.shape[0], desc='ims'):
            fn = '/' + image_id
            y_val[i] = np.array(f.get_node(fn))[..., None]
            
    fn_im = FMT_VALTEST_IM_FOLDER if is_test else FMT_VALTRAIN_IM_FOLDER
    X_val = np.empty((slice_n if debug else df_train.shape[0], ORIGINAL_SIZE, ORIGINAL_SIZE, 3))
    if max_workers == 1:
        for i, image_id in tqdm.tqdm_notebook(enumerate(df_train.ImageId.tolist()[:slice_n]),\
		total=df_train.shape[0], desc='ims'):
            X_val[i] = plt.imread(fn_im + image_id + '.png')[...,:3]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as e:
            gen = e.map(plt.imread, [fn_im + image_id + '.png'\
                    for image_id in df_train.ImageId.tolist()[:slice_n]])
            for i, im in enumerate(gen):
                X_val[i] = im[...,:3]
#         im = np.moveaxis(im, -1, 0)

    X_val, y_val = X_val.astype('float'), y_val.astype('float')
    return X_val, y_val

def get_dataset(datapath, debug=False):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    trn_x, trn_y = get_data(area_id, False, debug=debug)
    trn_y = np.broadcast_to(trn_y, [trn_y.shape[0], ORIGINAL_SIZE, ORIGINAL_SIZE, 3])
    val_x, val_y = get_data(area_id, True, debug=debug)
    val_y = np.broadcast_to(val_y, [val_y.shape[0], ORIGINAL_SIZE, ORIGINAL_SIZE, 3])
    return (trn_x,trn_y), (val_x,val_y)

class ArraysSingleDataset(BaseDataset):
    def __init__(self, x, y, transform, num_slice=9, sz=192):
        # input: ch x w x h
        self.x, self.y = x, y
        self.sz_i = self.x[0].shape[1]
        self.sz = sz
        self.num_slice = num_slice
        super().__init__(transform)
        
    def get_im(self, i, is_y):
        if is_y:
            im = self.y[i//self.num_slice]
        else:
            im = self.x[i//self.num_slice]
        slice_pos = i % self.num_slice
        a = np.sqrt(self.num_slice)
        cut_i = slice_pos // a
        cut_j = slice_pos % a
        stride = (self.sz_i - self.sz) // a
        cut_x = int(cut_j * stride)
        cut_y = int(cut_i * stride)
        return im[cut_x:cut_x + self.sz, cut_y:cut_y + self.sz]
    
    def get_x(self, i): return self.get_im(i, False)
    def get_y(self, i): return self.get_im(i, True)
    def get_n(self): return self.x.shape[0] * self.num_slice
    def get_sz(self): return self.sz
    def get_c(self): return 1
    def denorm(self, arr):
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
#         return np.clip(self.transform.denorm(np.rollaxis(arr,1,4)), 0, 1)
        return self.transform.denorm(np.rollaxis(arr,1,4))

(trn_x,trn_y), (val_x,val_y) = (None, None), (None, None)
class PublicArrayDataset(BaseDataset):
    def __init__(self, is_trn, y, transform, num_slice=9, sz=192):
        self.is_trn = is_trn
        self.sz_i = trn_x[0].shape[1] if self.is_trn else val_x[0].shape[1]
        self.sz = sz
        self.num_slice = num_slice
        super().__init__(transform)
    def get_im(self, i, is_y):
        if is_y:
            im = trn_y[i//self.num_slice] if self.is_trn else val_y[i//self.num_slice]
        else:
            im = trn_x[i//self.num_slice] if self.is_trn else val_x[i//self.num_slice]
        slice_pos = i % self.num_slice
        a = np.sqrt(self.num_slice)
        cut_i = slice_pos // a
        cut_j = slice_pos % a
        stride = (self.sz_i - self.sz) // a
        cut_x = int(cut_j * stride)
        cut_y = int(cut_i * stride)
        return im[cut_x:cut_x + self.sz, cut_y:cut_y + self.sz]

    def get_x(self, i): return self.get_im(i, False)
    def get_y(self, i): return self.get_im(i, True)
    def get_n(self): return trn_x.shape[0] * self.num_slice if self.is_trn else val_x.shape[0] * self.num_slice
    def get_sz(self): return self.sz
    def get_c(self): return 1
    def denorm(self, arr):
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
#         return np.clip(self.transform.denorm(np.rollaxis(arr,1,4)), 0, 1)
        return self.transform.denorm(np.rollaxis(arr,1,4))

class UpsampleModel():
    def __init__(self, model, cut_base=8, name='upsample'):
        self.model,self.name = model,name
        self.cut_base = cut_base

    def get_layer_groups(self, precompute):
        c = children(self.model.module)
        return [c[:self.cut_base],
           c[self.cut_base:]]

def sep_iou(y_pred, y_true, thresh=0.5):
    return np.array([jaccard_coef(p, t) for (p, t) in zip(y_pred, y_true)])
    
## cuda version
def jaccard_coef_cuda(y_pred, y_true, thresh=0.5):
    smooth = T(1e-12)
    y_pred = (y_pred > thresh).float()
    y_true = (y_true > thresh).float()
    intersection = y_true * y_pred
    sum_ = torch.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return torch.mean(jac)

## np version
def jaccard_coef(y_pred, y_true=None, thresh=0.5):
    if isinstance(y_pred, tuple):
        y_pred, y_true = y_pred
    elif y_true is None:
        raise TypeError
        
    smooth = 1e-12
    y_pred = to_np(y_pred) > thresh
    y_true = to_np(y_true) > thresh
    intersection = y_true * y_pred
    sum_ = np.sum(y_true) + np.sum(y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return np.mean(jac)

def sep_parallel(f, y_pred, y_true, num_workers=8, **kwargs):
    scores = np.empty(y_true.shape[0])
    if num_workers == 0:
        for i, (p, y) in enumerate(zip(y_pred, y_true)):
            scores[i] = f(p, y, **kwargs)
        return scores
        
    with ThreadPoolExecutor(max_workers=num_workers) as e:
        res = e.map(partial(f, **kwargs), y_pred, y_true)
        for i, score in enumerate(res):
            scores[i] = score
        return scores
    
def jaccard_coef_parallel(y_pred, y_true, thresh=0.5, num_workers=8):
    if num_workers == 0:
        return jaccard_coef(y_pred, y_true, thresh=thresh)
    with ThreadPoolExecutor(max_workers=num_workers) as e:
        jac = list(e.map(partial(jaccard_coef, thresh=thresh), zip(y_pred, y_true)))
        return np.mean(jac)

def get_rgb_mean_stat(area_id):
    prefix = area_id_to_prefix(area_id)

    with tb.open_file(FMT_IMMEAN.format(prefix), 'r') as f:
        im_mean = np.array(f.get_node('/immean'))[:3]
    
    mean = [np.mean(im_mean[i]) for i in range(3)]
    std = [np.std(im_mean[i]) for i in range(3)]
    return np.stack([np.array(mean), np.array(std)])

def get_md_model(datapaths, data, bs, device_ids, num_workers, model_name='unet', global_dataset=False, num_slice=9, sz=192):
#     (trn_x, trn_y), (val_x, val_y) = trn, val
    if global_dataset:
        global trn_x,trn_y, val_x,val_y
        
    aug_tfms = transforms_top_down
    for o in aug_tfms: o.tfm_y = TfmType.CLASS
        
    area_ids = [directory_name_to_area_id(datapath) for datapath in datapaths]
    stats = np.mean([get_rgb_mean_stat(area_id) for area_id in area_ids], axis=0)
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
    
    (trn_x,trn_y), (val_x,val_y) = data
    dataset = PublicArrayDataset if global_dataset else ArraysSingleDataset
    trn, val = ((True, True), (True, True)) if global_dataset else ((trn_x,trn_y), (val_x,val_y))
    datasets = ImageData.get_ds(dataset, trn, val, tfms, num_slice=num_slice, sz=sz)
    md = ImageData('data', datasets, bs, num_workers=num_workers, classes=None)
    denorm = md.trn_ds.denorm

    if not Path(MODEL_DIR).exists():
        Path(MODEL_DIR).mkdir(parents=True)

    if model_name == 'deeplab':
        model = deeplab_resnet.Res_Deeplab(1) # change to 1
        cut_base = 1
    elif model_name == 'deeplab2':
        model = deeplab_resnet.Res_Deeplab(2) # change to 1
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

def expanded_loss(pred, target):
    return F.binary_cross_entropy(pred[:,0], target)
def mask_acc(pred,targ): return accuracy_multi(pred[:,0], targ, 0.)
def get_learn(md, model):
    learn=ConvLearner(md, model)
    learn.opt_fn=optim.Adam
    learn.crit=crit
    learn.metrics=[mask_acc]
    return learn
    
def learner_on_dataset(datapath, bs, device_ids, num_workers, model_name='unet', debug=False,
        data=None, global_dataset=False, num_slice=9, sz=192):
    if data is None:
        data = (trn_x,trn_y), (val_x,val_y) = get_dataset(datapath, debug=debug)
    md, model, denorm = get_md_model([datapath], data, bs, device_ids,
            num_workers, model_name=model_name, global_dataset=global_dataset,
            num_slice=num_slice, sz=sz)
    print('Data finished loading:', datapath)
    learn = get_learn(md, model)
    return learn, denorm, data

#def load_backup_learn(datapath, data, bs, device_ids, num_workers, model_name='unet', global_dataset=False):
#   md, model, denorm = get_md_model([datapath], data, bs, device_ids, num_workers, model_name=model_name, global_dataset=global_dataset)
#   learn = get_learn(md, model)
#   return learn, denorm

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
    
def bool_pred(pred, thresh=0.5):
    return to_np(pred > thresh)

def val_loss(preds, y, loss, thresh=None):
    if thresh is not None:
        preds = bool_pred(preds, thresh)
    return loss(preds, y)

def plot_worse_cross_entropy(tta, shift=0, n_ims=9, is_best=False, step=2):
    pass

def plot_ims(data, labels=None):
    # data and labels should be zips
    data = list(data)
    n_ims = len(data)
    cols = len(list(zip(*data)))
    if labels is not None:
        labels = list(labels)
    else:
        labels = np.zeros((n_ims, cols))
    fig, ax = plt.subplots(n_ims, cols, figsize=(3*cols, 3*n_ims))
    for i, row in enumerate(ax):
        if len(ax.shape) == 1:
            row.imshow(data[0][i])
            row.set_xlabel(labels[0][i])
        else:
            for j, col in enumerate(row):
                col.imshow(data[i][j])
                col.set_xlabel(labels[i][j])
    fig.tight_layout()

def plot_worse_preds(x, y, preds, learn, crit=jaccard_coef, scores=None, shift=0, n_ims=9,
        is_best=False, thresh=0.5, **kwargs):
    if scores is None:
        scores = sep_parallel(crit, T(preds), T(y), **kwargs)
    lowest_iou_idx = np.argsort(scores)
    if is_best:
        lowest_iou_idx = np.flip(lowest_iou_idx, 0)
    lowest_iou_idx = lowest_iou_idx[shift: n_ims + shift]
    labels = [["x"] * n_ims, 
        ["gt"] * n_ims,
        scores[lowest_iou_idx],
        ["preds"] * n_ims]
    bp = bool_pred(preds, thresh=thresh)
    xl, yl, predsl, bpl = [], [], [], []
    for i in lowest_iou_idx:
        xl.append(learn.data.trn_ds.denorm(x[i]).squeeze())
        yl.append(y[i])
        predsl.append(preds[i])
        bpl.append(bp[i])
    data = zip(xl, yl, predsl, bpl)    
    labels = zip(*labels)
    plot_ims(data, labels=labels)
    return scores, lowest_iou_idx

# sequential: if True, in one outer loop, every dataset is trained only once
def train_on_full_dataset(epochs, lrs, wds, sequential=False, save_starter='full_dataset_beginner',\
                          cycle_len=2, cycle_mult=2, save_path=Path('data/figs'), datapath_slice=0,\
                         epoch_shift=0, use_wd_sched=False, model_name='unet', **kwargs):
    global learn, denorm
    for out_epoch in tqdm.tnrange(epochs if sequential else 1, desc='out'):
        datapath_slice = 0 if sequential else datapath_slice
        for i, datapath in tqdm.tqdm_notebook(enumerate(datapaths[datapath_slice:]),\
                                              total=len(datapaths), desc='datapaths'):
            i += epoch_shift
            if last_datapath == datapath:
                learn, denorm = load_backup_learn(model_name=model_name)
            else:
                learn, denorm = learner_on_dataset(datapath, model_name=model_name)
            
            best_save_name = 'full_dataset_out' if sequential else 'full_dataset_in'
            epoch_save_name_base = 'full_dataset'
            epoch_save_name_base += '_out_' if sequential else '_in_'
            
            if out_epoch:
                learn.load(epoch_save_name_base + str(out_epoch - 1))
            elif save_starter != '':
                learn.load(save_starter)
            
            learn.unfreeze();
            in_epochs = epochs if not sequential else 1
            learn.fit(lrs, in_epochs, wds=wds, use_wd_sched=use_wd_sched, cycle_len=cycle_len,\
                            cycle_mult=cycle_mult, use_clr=None,\
                            best_save_name=best_save_name, **kwargs)

            learn.save(epoch_save_name_base + str(out_epoch))
            
            save_name = epoch_save_name_base + str(out_epoch) + '_' + str(i)
            plot_lr_loss(learn, Path(save_name))

crit = expanded_loss
metrics = jaccard_coef_parallel

docstr = """
Usage:
    learn.py [options]

Options:
    -h, --help                  Print this message
    
    Read code for more.
"""
if __name__ == '__main__':
    args = docopt(docstr)
    print(args)
    num_gpus = args['--num_gpus']
    gpu_start = args['--gpu_start']
    torch.cuda.set_device(gpu_start)
    num_workers = 3 * num_gpus
    device_ids = range(gpu_start, gpu_start + num_gpus)
    torch.cuda.set_device(gpu_start)
    bs = args['--bs'] * num_gpus 

    opt = args['--opt']
    epochs = args['--epochs']
    lr = args['--lr']
    wd = args['--wd']
    use_wd_sched = args['--uws']
    lr_div = args['--lr_div']
    save_starter = args['--save_starter']
    model_name = args['--model_name']
    debug = args['--debug']

    if opt == 'full':
        sequential = args['--sequential']
        train_on_full_dataset(epochs=epochs, lrs=[lr/lr_div, lr], sequential=sequential, wds=[wd/lr_div, wd],
                use_wd_sched=use_wd_sched, save_starter='full_dataset_in_0', model_name=model_name)
    else:
        city_code = args['city_code']
        learn, denorm = learner_on_dataset(datapaths[city_code])

        save_idx = args['save_idx']
        save_name = args['save_name']
        cycle_len = args['cycle_len']
        train_and_plot(save_idx, save_name, epoch=epochs, lrs=lrs, wds=wds, use_wd_sched=use_wd_sched,
                cycle_len=cycle_len, use_clr=None,
                best_save_name=save_name)

def learn_init(num_gpus=1, gpu_start=0, bs=32, debug=False):
    global _num_gpus, _gpu_start, _num_workers, _device_ids, _bs, _debug
    _num_gpus = num_gpus
    _gpu_start = gpu_start
    _num_workers = 3 * num_gpus
    device_ids = range(gpu_start, gpu_start + num_gpus)
    _bs = 32 * num_gpus 
    _debug = debug
    torch.cuda.set_device(_gpu_start)

