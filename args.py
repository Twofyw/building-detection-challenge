import argparse
helpstr = """ Accepts an datapath index. Cities are arranged in the order of:
'data/train/AOI_2_Vegas_Train', 'data/train/AOI_3_Paris_Train', 
             'data/train/AOI_4_Shanghai_Train', 'data/train/AOI_5_Khartoum_Train'

"""
parser = argparse.ArgumentParser(description=helpstr)
parser.add_argument('--datapath_idx', type=int, help='the datapath index')
parser.add_argument('--pred', help='prediction trigger')
parser.add_argument('--eval', help='evaluation trigger')
parser.add_argument('--test', help='test trigger')
parser.add_argument('--debug', help='load only first 50 images', action='store_true', default=False)

args = parser.parse_args()


if 'hidden' in globals():
    print('global')
if 'hidden' in locals():
    print('local')
hidden = 'found'