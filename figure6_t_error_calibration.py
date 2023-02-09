import os
import matplotlib.pyplot as plt
import pickle

dir_name = '/mp/nas2/ruihan/multi_view_pose_refine/output05/loss_weight_rgb_0.10_ssim_ms_0.10_perceptual_0.80/'

dir_name = '/mp/nas2/ruihan/multi_view_pose_refine/output01/loss_weight_0.40_0.30_0.30/'

dir_name = '/mp/nas2/ruihan/multi_view_pose_refine/output01/loss_weight_rgb_0.20_ssim_ms_0.40_perceptual_0.40/'

dir_name = '/mp/nas2/ruihan/multi_view_pose_refine/output09/loss_weight_rgb_0.20_ssim_ms_0.70_perceptual_0.10/'

dir_name = '/mp/nas2/ruihan/multi_view_pose_refine/output01/converge_rgb_1.00_ssim_ms_0.20_perceptual_1.00/'

# for file in os.listdir(dir_name):
#     if 'basin' in file:
#         continue
#     with open(dir_name + file,'rb') as f:
#         data = pickle.load(f)
#     # print(data['losses']['rgb'].cpu().detach().numpy())
#     # plt.plot(data['losses']['rgb'].cpu().detach().numpy())
#     # plt.plot(data['t_error'])
#     plt.imshow(data['img'])
#     plt.savefig('plots/figure6/{}.jpg'.format(data['t_error'][-1]))
# # plt.savefig('plots/object9_converge_t_errors.jpg')

plt.close()
for file in os.listdir(dir_name):
    if 'basin' in file:
        continue
    with open(dir_name + file,'rb') as f:
        data = pickle.load(f)
    plt.plot(data['t_error'])
    # plt.plot([x.cpu().detach() for x in data['losses_dict']['rgb']])
plt.savefig('plots/object9_converge_t_errors.jpg')