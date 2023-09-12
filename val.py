import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from archs import UNext
from models_custom.unet_mobilevig import UNetMobileVig
from models_custom.visual import visulize_feature, SaveOutput

from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
# import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image)
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])
    model = UNetMobileVig(local_channels=[32, 64, 128, 256], global_channels=512, drop_path=0.1)

    model = model.cpu()

    # Data loading code
    config['img_ext'] = '.png'
    img_ids = glob(os.path.join('inputs', "test", 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name'], map_location='cpu'))
    model.to('cpu')
    model.eval()

    img = np.array(Image.open('./demo.jpg').convert('RGB'))
    img = cv2.resize(img, (224, 224))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    import torchvision
    transform = torchvision.transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)

    # target_layers = [model.up_conv_stage4]
   
    # cam = EigenCAM(model, target_layers, use_cuda=False)
    # grayscale_cam = cam(tensor)[0, :, :]
    # cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    # test = Image.fromarray(cam_image)
    # test.save(f'test.jpg')



    # ##### pytorch hooks #####
    # save_output = SaveOutput()
    # hook_handles = []
    # for layer in model.modules():
    #     if isinstance(layer, torch.nn.modules.conv.Conv2d):
    #         handle = layer.register_forward_hook(save_output)
    #         hook_handles.append(handle)
   
    # output = model(tensor)

    # def module_output_to_numpy(tensor):
    #     return tensor.detach().to('cpu').numpy()    

    # images = module_output_to_numpy(save_output.outputs[0])
    # print(images.shape)
    # for i in range(len(images[0])):
    #     cv2.imwrite(f'./feature_{i}.jpg', deprocess_image(images[0,i]))



    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])



    config['dataset'] = "test"
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config['batch_size'],
        batch_size= 1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta, img_origin in tqdm(val_loader, total=len(val_loader)):
            input = input.cpu()
            target = target.cpu()
            model = model.cpu()
            # compute output
            output = model(input)


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    # cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                    #             (output[i, c] * 255).astype('uint8'))

                    # # concate 3 image into 1
                    # put the text in every image
                    target_img = cv2.cvtColor((target[i, c].cpu().numpy() * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
                    mask_img = cv2.cvtColor((output[i, c] * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
                    img_origin = cv2.resize(img_origin[i].numpy(), (224, 224))

                    # Find contours
                    contours, hierarchy = cv2.findContours((output[i, c] * 255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    img_origin = cv2.drawContours(img_origin, contours, -1, (0, 0, 255), 1)
                    
                    contours1, hierarchy = cv2.findContours((target[i, c].cpu().numpy() * 255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    img_origin = cv2.drawContours(img_origin, contours1, -1, (0, 255, 0), 1)
                    img_concate = cv2.hconcat([img_origin, target_img, mask_img])

                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '_result' + '.jpg'),img_concate)
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

# python train.py --dataset isic2017 --arch UNext  --img_ext .jpg --mask_ext .jpg --lr 0.0001 --epochs 500 --input_w 512 --input_h 512 --b 8