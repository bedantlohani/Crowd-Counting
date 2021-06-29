import os
import torch
import numpy as np
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils
import pandas as pd


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

#test data and model file path
data_path =  '/home/bedant/crowdcount-cascaded-mtl/eval'
gt_path = '/home/bedant/crowdcount-cascaded-mtl/output/gt'
model_b = '/home/bedant/crowdcount-cascaded-mtl/final_models/cmtl_shtechB_768 (1).h5'
model_a = '/home/bedant/crowdcount-cascaded-mtl/final_models/cmtl_shtechA_204.h5'
model_path = model_a


output_dir = '/home/bedant/crowdcount-cascaded-mtl/output/results'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

net = CrowdCounter()

trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
mae = 0.0
mse = 0.0
name = [img_name for img_name in os.listdir(data_path)]
y_true = []
y_pred = []
for blob in data_loader:
    im_data = blob['data']
    gt_data = blob['gt_density']
    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)
    y_true.append(gt_count)
    y_pred.append(et_count)
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        utils.save_results(im_data, gt_data, density_map, output_dir +'test_results_img/test_results_imgB', blob['fname'].split('.')[0] + '.png')


data = pd.DataFrame({'name': name,'y_pred': y_pred,'y_true': y_true})
data.to_csv('CSV/on_A_test_CMTL.csv', sep=',')

data = pd.read_csv('CSV/on_A_test_CMTL.csv')
y_true = data['y_true']
y_pred = data['y_pred']
mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print ('MAE: %0.2f, MSE: %0.2f' % (mae,mse))

f = open(file_results, 'w')
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()
