import argparse
helpstr = """ Accepts an datapath index. Cities are arranged in the order of:
'data/train/AOI_2_Vegas_Train', 'data/train/AOI_3_Paris_Train', 
             'data/train/AOI_4_Shanghai_Train', 'data/train/AOI_5_Khartoum_Train'

"""
parser = argparse.ArgumentParser(description=helpstr)
parser.add_argument('datapath_idx', type=int, help='the datapath index')
parser.add_argument('--num_gpus', type=int, help='number of GPU cores', default=1)
parser.add_argument('--gpu_start', type=int, help='first GPU index', default=0)
parser.add_argument('--bs', type=int, help='batch size', default=8)
parser.add_argument('--num_slice', type=int, help='number of slices to cut', default=9)

# train
parser.add_argument('--train', help='training trigger', action='store_true', default=False)
parser.add_argument('--rescale', help='rescaling trigger', action='store_true', default=False)
parser.add_argument('--lr', type=float, help='set lr')
parser.add_argument('--use_wd_sched', help='training trigger', action='store_true', default=False)
parser.add_argument('--wd', type=float, help='set wd')
parser.add_argument('--n_cycles', type=int, help='number of epochs')


# predict
parser.add_argument('--pred', help='prediction trigger', action='store_true', default=False)
parser.add_argument('--eval', help='evaluation trigger', action='store_true', default=False)
parser.add_argument('--test', help='test trigger', action='store_true', default=False)
parser.add_argument('--debug', help='load only first 50 images', action='store_true', default=False)
args = parser.parse_args()
if args.eval: args.pred = True

    
    
# Args parsing end
from learn import *
sys.path.insert(0, 'code')
from v17 import *
from v17 import _internal_validate_predict_best_param

sz = 256
num_slice = args.num_slice

num_gpus = args.num_gpus
gpu_start = args.gpu_start
num_workers = 8
device_ids = range(gpu_start, gpu_start + num_gpus)
# device_ids = [0,1,4,5]
torch.cuda.set_device(gpu_start)
bs = args.bs

model_name = 'deeplab'
datapaths = ['data/train/AOI_2_Vegas_Train', 'data/train/AOI_3_Paris_Train', 
             'data/train/AOI_4_Shanghai_Train', 'data/train/AOI_5_Khartoum_Train']
model_name = 'deeplab'
loadpaths = [model_name + '-' + o for o in ['vegas', 'paris', 'shanghai', '5']]
datapath = datapaths[args.datapath_idx]
loadpath = loadpaths[args.datapath_idx])

data = None
if args.train:
    is_eval, is_pred = False, False
    learn, denorm, data = learner_on_dataset(datapath, bs, device_ids, num_workers, model_name=model_name,
                                             debug=args.debug, data=data, num_slice=num_slice, sz=sz,
                                             is_eval=is_eval, is_pred=is_pred, rescale=args.rescale)
    learn.load(
    learn.unfreeze()

    lr = args.lr
    wd = args.wd
    use_wd_sched = args.use_wd_sched
    n_cycles = args.n_cycles

    train_and_plot(learn, 0, load_path, lrs = lr, n_cycles=n_cycles, wds=wds, use_wd_sched=use_wd_sched,
        cycle_len=2, cycle_mult=2, best_save_name=load_path)

if args.pred:
    is_eval, is_pred = True, False 
    learn, denorm, data = learner_on_dataset(datapath, bs, device_ids, num_workers, model_name=model_name,
                                             debug=args.debug, data=data, num_slice=num_slice, sz=sz,
                                            is_eval=is_eval, is_pred=is_pred)
    learn.load(loadpath)
    preds = learn.predict().squeeze()
    
    if args.eval:
        evalfscore(datapath, preds)
        
if args.test:
    is_eval, is_pred = False, True
    learn, denorm, data = learner_on_dataset(datapath, bs, device_ids, num_workers, model_name=model_name,
                                             debug=args.debug, data=data, num_slice=num_slice, sz=sz,
                                            is_eval=is_eval, is_pred=is_pred)
    learn.load(loadpath)
    preds = learn.predict().squeeze()










