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
parser.add_argument('--num_slice', type=int, help='number of slices to cut per image', default=9)
parser.add_argument('--model_name', help='unet/deeplab', default='deeplab')
parser.add_argument('--sz', type=int, help='sz', default=256)
parser.add_argument('--num_workers', type=int, help='number of workers', default=8)
parser.add_argument('--noload', help='if there is no model to start with', action='store_true', default=False)

# train
parser.add_argument('--learn', help='learning trigger', action='store_true', default=False)
parser.add_argument('--rescale', help='rescaling trigger', action='store_true', default=False)
parser.add_argument('--lr', type=float, help='set lr')
parser.add_argument('--use_wd_sched', help='training trigger', action='store_true', default=False)
parser.add_argument('--wd', type=float, help='set wd')
parser.add_argument('--n_cycles', type=int, help='number of epochs')
parser.add_argument('--load_starter', type=str, help='begin training with this model', default='')
parser.add_argument('--start', help='start immediately', action='store_true', default=False)


# predict
parser.add_argument('--pred', help='prediction trigger', action='store_true', default=False)
parser.add_argument('--eval', help='evaluation trigger', action='store_true', default=False)
parser.add_argument('--test', help='test trigger', action='store_true', default=False)
parser.add_argument('--debug', help='load only first 50 images', action='store_true', default=False)
args = parser.parse_args()
if args.eval: args.pred = True

    
    
# Args parsing end
from lib import *
sys.path.insert(0, 'code')
from v17 import *

sz = args.sz
num_slice = args.num_slice

num_gpus = args.num_gpus
gpu_start = args.gpu_start
num_workers = args.num_workers
device_ids = range(gpu_start, gpu_start + num_gpus)
# device_ids = [0,1,4,5]
torch.cuda.set_device(gpu_start)
bs = args.bs

model_name = args.model_name
datapaths = ['data/train/AOI_2_Vegas_Train', 'data/train/AOI_3_Paris_Train', 
             'data/train/AOI_4_Shanghai_Train', 'data/train/AOI_5_Khartoum_Train']
load_paths = [model_name + '-' + o for o in ['vegas', 'paris', 'shanghai', '5']]
datapath = datapaths[args.datapath_idx]
base_load_path = load_paths[args.datapath_idx]
base_save_path = 'debug' if args.debug else base_load_path

scale_load_path = base_load_path + '-scale'
scale_save_path = base_save_path + '-scale'





def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def manual_predict(learn):
    n = learn.data.val_ds.get_n()
    preds = np.empty((n, INPUT_SIZE, INPUT_SIZE))
    for o in tqdm.tqdm_notebook(range(0, n, bs)):
        end = min(n, o + bs)
        X = np.moveaxis(learn.data.val_ds.transform.norm([learn.data.val_ds.get_x(o) for o in range(o, end)])[0], -1, 1)
        preds[o:end] = sigmoid(learn.predict_array(X).squeeze())
    preds = skimage.transform.resize(preds, (650, 650))
    return preds
 





data = None
################################## train
if args.learn:
    is_eval, is_pred = False, False
    learn, denorm, data = learner_on_dataset(datapath, bs, device_ids, num_workers, model_name=model_name,
                                             debug=args.debug, data=data, num_slice=num_slice, sz=sz,
                                             is_eval=is_eval, is_pred=is_pred, rescale=args.rescale)
    if args.rescale:
        learn_load_path = scale_load_path
        learn_save_path = scale_save_path
    else:
        learn_load_path = base_load_path
        learn_save_path = base_save_path

    if args.load_starter != '':
        learn_load_path = args.load_starter
    if not args.noload:
        learn.load(learn_load_path)
    learn.unfreeze()

    lr = args.lr
    lrs = [lr / 9, lr]
    wd = args.wd
    wds = [wd / 3, wd]
    use_wd_sched = args.use_wd_sched
    n_cycles = args.n_cycles

    if args.start:
        train_and_plot(learn, 0, learn_save_path, lrs = lrs, n_cycles=n_cycles, wds=wds, use_wd_sched=use_wd_sched,
            cycle_len=2, cycle_mult=2, best_save_name=learn_save_path)




################################### pred
elif args.pred:
       
    is_eval, is_pred = True, False 
    # use crop
    learn, denorm, data = learner_on_dataset(datapath, bs, device_ids, num_workers, model_name=model_name,
                                             debug=args.debug, data=data, num_slice=num_slice, sz=sz,
                                            is_eval=is_eval, is_pred=is_pred, rescale=False)
    if args.load_starter != '':
        base_load_path = args.load_starter
    learn.load(base_load_path)
    learn.model.eval()
    preds_crop = []
    preds_scale = [manual_predict(learn)]

    # use rescale
    #if args.rescale:
    #    pred_load_path = base_load_path + '-scale'

    #    learn, denorm, data = learner_on_dataset(datapath, bs, device_ids, num_workers, model_name=model_name,
    #                                         debug=args.debug, data=data, num_slice=num_slice, sz=sz,
    #                                        is_eval=is_eval, is_pred=is_pred, rescale=True,
    #                                        )
    #    learn.load(pred_load_path)
    #    preds_rescale = [manual_predict(learn)]
    #else:
    #    preds_rescale = []

    if args.eval:
        # find optimum threshold
        fscores, prs, y_pred = evalfscore(datapath, preds_crop, preds_scale, 
                debug=args.debug, num_slice=num_slice)
        # fscores is automatically imported into jupyter notebook
        





###################################### test
elif args.test:
    is_eval, is_pred = False, True
    learn, denorm, data = learner_on_dataset(datapath, bs, device_ids, num_workers, model_name=model_name,
                                             debug=args.debug, data=data, num_slice=num_slice, sz=sz,
                                            is_eval=is_eval, is_pred=is_pred)
    if args.load_starter != '':
        base_load_path = args.load_starter
    learn.load(base_load_path)
    # predict cropped only once
    y_preds = [manual_predict(learn)]
    testproc(datapath, y_preds, num_slice=num_slice)

    







