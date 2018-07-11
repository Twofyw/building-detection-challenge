from fastai.dataset import *
from fastai.transforms import *
from fastai.conv_learner import *

sys.path.insert(0, 'code')
from v17 import *
from v17 import _internal_pred_to_poly_file
from tqdm import tqdm_notebook
from models import *
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from IPython.core import debugger
from functools import partial

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--area_id', type=int)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_slice', type=int, default=25)
parser.add_argument('--gpu_start', type=int, default=0)
parser.add_argument('--num_gpus', type=int, default=1)
args = parser.parse_args()

area_id       = args.area_id
area_id_idx   = area_id - 2
prefix        = area_id_to_prefix(area_id)
ORIGINAL_SIZE = 650
sz            = 256
num_workers   = args.num_workers
num_slice     = args.num_slice
bs            = args.bs

padding_sz    = 59
padded_sz     = ORIGINAL_SIZE + 2 * padding_sz

gpu_start     = args.gpu_start
num_gpus      = args.num_gpus
device_ids = range(gpu_start, gpu_start + num_gpus)
torch.cuda.set_device(gpu_start)

PATH                    = Path('data')
DATA_PATH               = Path('working/images/v5')
TRN                     = Path('trn_test_full_rgb')
TEST                    = Path('pred_full_rgb')
FMT_VALTRAIN_FN         = '{}_valtrain_ImageId.csv'.format(prefix)
FMT_VALTEST_FN          = '{}_valtest_ImageId.csv'.format(prefix)
FMT_VALTRAIN_MASK_STORE = 'valtrain_{}_mask.h5'.format(prefix)
FMT_VALTEST_MASK_STORE  = 'valtest_{}_mask.h5'.format(prefix)
FMT_IMMEAN              = '{}_immean.h5'.format(prefix)
FMT_VALMODEL_EVALTHHIST = '{}_val_evalhist_th.csv'.format(prefix)
FMT_TEST_FN             = '{}_test_ImageId.csv'.format(prefix)

datapaths = ['data/train/AOI_2_Vegas_Train', 'data/train/AOI_3_Paris_Train',
             'data/train/AOI_4_Shanghai_Train', 'data/train/AOI_5_Khartoum_Train']
datapath = datapaths[area_id_idx]


model_name              = 'unet'
weight_name             = [model_name + '-' + o for o in ['vegas', 'paris', 'shanghai', '5']]
weight_load             = weight_name[area_id_idx]



class OptionalFilesArrayDataset(FilesArrayDataset):
    def __init__(self, fnames, y, transform, path, num_slice, pad):
        self.side = int(np.sqrt(num_slice))
        assert(self.side**2 == num_slice and num_slice >= 9)
        self.is_empty = (fnames == 'empty')

        if not self.is_empty:
            self.num_slice = num_slice
            self.pad = pad
            self.n = len(fnames) * num_slice
            if self.pad and y is not None and len(y.shape) > 2:
                padding_shape = [[0, 0], [padding_sz]*2, [padding_sz]*2, [0, 0]]
                y = np.pad(y, padding_shape, 'reflect')
            self.side_length = padded_sz if pad else ORIGINAL_SIZE
            super().__init__(fnames, y, transform, path)
        else:
            self.n = 0

    def switch_padding(on=True):
        if on and not self.pad:
            self.pad = True
            padding_shape = [[0, 0], [padding_sz]*2, [padding_sz]*2, [0, 0]]
            self.y = np.pad(self.y, padding_shape, 'reflect')
            self.side_length = padded_sz
        elif not on and self.pad:
            self.pad = False
            self.y = crop_center(self.y, padding_sz)
            self.side_length = ORIGINAL_SIZE

    def cut_im_at_pos(self, im, slice_pos):
        slice_pos %= self.num_slice
        cut_j, cut_i = divmod(slice_pos, self.side)
        stride = (self.side_length - sz) // (self.side - 1)
        cut_x = int(cut_j * stride)
        cut_y = int(cut_i * stride)
        return im[cut_x:cut_x + sz, cut_y:cut_y + sz]
        
    def get_x(self, i):
        I = i // self.num_slice
        fn = self.path/self.fnames[I]
        if self.pad:
            x = self.read_cached_im(fn, True)
        else:
            x = self.read_cached_im(fn, False)
        return self.cut_im_at_pos(x, i)
    
    def get_y(self, i):
        I = i // self.num_slice
        if len(self.y.shape) == 2:
            return self.y[I, None]
        y = self.y[I]
        return self.cut_im_at_pos(y, i)
    
    def get_c(self): return 1
    def get_n(self): return self.n
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def read_cached_im(fn, pad):
        if pad:
            padding_shape = [[padding_sz]*2, [padding_sz]*2, [0, 0]]
            im = open_image(fn)
            return np.pad(im, padding_shape, 'reflect')
        else:
            return open_image(fn)

def load_y(image_ids, fn):
    n = len(image_ids)
    y = np.empty((n, ORIGINAL_SIZE, ORIGINAL_SIZE, 1))

    with tb.open_file(str(PATH/DATA_PATH/fn)) as f:
        for i, image_id in tqdm_notebook(enumerate(image_ids), total=n):
            fn = '/' + image_id
            y[i] = np.array(f.get_node(fn))[..., None]
    y = np.broadcast_to(y, (n, ORIGINAL_SIZE, ORIGINAL_SIZE, 3))
    return y

def get_model():
    model = to_gpu(UNet16(pretrained='vgg'))
    cut_base = 8
    model = nn.DataParallel(model, device_ids=device_ids)
    model = UpsampleModel(model, cut_base=cut_base)
    return model

def get_learn(md, model, load_weight=False):
    learn = ConvLearner(md, model)
    learn.opt_fn=optim.Adam
    learn.crit=expanded_loss
    learn.metrics=[mask_acc, jaccard_coef_par]
    if load_weight:
        learn.load(weight_load)
    return learn


class UpsampleModel():
    def __init__(self, model, cut_base=8, name='upsample'):
        self.model,self.name = model,name
        self.cut_base = cut_base

    def get_layer_groups(self, precompute):
        c = children(self.model.module)
        return [c[:self.cut_base],
                c[self.cut_base:]]

def expanded_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred[:,0], target) -\
            torch.log(jaccard_coef_loss(pred[:,0], target))

def jaccard_coef_loss(y_pred, y_true):
    epsilon = 1e-12
    y_pred_pos = torch.clamp(y_pred, 0, 1)
    intersection = torch.sum(y_true * y_pred_pos)
    sum_ = torch.sum(y_true + y_pred_pos)
    return (intersection + epsilon) / (sum_ - intersection + epsilon)

def mask_acc(pred,targ): return accuracy_multi(pred[:,0], targ, 0.)

def jaccard_coef_par(pred, target):
    return jaccard_coef_loss(pred[:,0], target)


def get_rgb_mean_stat(area_id):
    prefix = area_id_to_prefix(area_id)

    with tb.open_file(str(PATH/DATA_PATH/FMT_IMMEAN), 'r') as f:
        im_mean = np.array(f.get_node('/immean'))[:3]
    mean = im_mean.mean(axis=(1, 2))
    std = imagenet_stats[1]
    return mean, std

def manual_predict(learn):
    pass        	














######################################## Plot
def plot_ims(data, labels=None, figsize=3):
    # data and labels should be zips
    data = [(o[0].squeeze(), o[1].squeeze()) for o in data]
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


def crop_center(im, padding_sz):
    return im[padding_sz:-padding_sz,
              padding_sz:-padding_sz]

def put_back_slices(ims, padded=True):
    # ims.shape --> (num_slice, length, hight, channels)
    side = int(np.sqrt(num_slice))
    assert(side**2 == num_slice and num_slice >= 9)
    assert(ims.shape[0] == num_slice and ims.shape[-1] < 10)
#     debugger.set_trace()
    side_length = padded_sz if padded else ORIGINAL_SIZE
    final_shape = [side_length, side_length, ims.shape[-1]]
    stride = (side_length - sz) / (side - 1)
    final = np.zeros(final_shape)
    count = np.zeros(final_shape[:2] + [1])
    for slice_pos, im in enumerate(ims):
        pos_j, pos_i = divmod(slice_pos, side)
        # naming error: x and y should be swapped
        # for numpy, y comes first
        # but for consistancy, ignore for now
        x = int(stride * pos_j)
        y = int(stride * pos_i)
        
#         debugger.set_trace()
        final[x:x+sz, y:y+sz] += im
        count[x:x+sz, y:y+sz] += 1
    #debugger.set_trace()
    final /= count
    if padded:
        final = crop_center(final, padding_sz)
    return final



def put_back_parallel(ims, padded=True, max_workers=num_workers):
    # Create batches
    ims = ims.squeeze()
    if ims.shape[-1] == sz:
        ims = ims[...,None]
    batches = np.split(ims, ims.shape[0] // num_slice)
    
    # No external calls. Use multiprocessing
#     return map(put_back_slices, batches)
#     with ProcessPoolExecutor(max_workers=num_workers) as e:
    with ThreadPoolExecutor(max_workers=max_workers) as e:
        return e.map(partial(put_back_slices, padded=padded), batches)




