import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image, make_uniform_heatmap
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2', help='Torch device to use')
    # parser.add_argument('--image-path', type=str, default='./examples/both.png', help='Input image path')
    parser.add_argument('--img-name', type=str, default='ILSVRC2012_val_00002100.JPEG', help='Input image name')
    parser.add_argument('--white-model', type=str, default='resnet18', help='White model')
    parser.add_argument('--attack', type=str, default='admix', help='Attack method')
    parser.add_argument('--target-model', type=str, default='resnet18', help='Target model')

    parser.add_argument('--aug-smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen-smooth', action='store_true', help='Reduce noise by taking the first principle component of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output', help='Output directory to save the images')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args

def find_index(target_name, file_path= '/home/shuxue2/yy/TransferAttackv2/new_data/labels.csv'):
                               # 请替换为你的CSV文件路径
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if row[0] == target_name:
                return int(row[1])  # 返回第二列的序号，转换为整数类型
    return -1

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    model_name = args.target_model
    # cnn_model_paper = ['resnet18', 'resnet101', 'resnext50_32x4d', 'densenet121']
    model = models.__dict__[model_name](weights="DEFAULT")
    model = model.to(torch.device(args.device)).eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50, 101: model.layer4
    # VGG, densenet161, desenet121: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    # print(model)
    target_layers = [model.layer4]
    #read adv_image
    img_path_adv = "/home/shuxue2/yy/TransferAttackv2/adv_data/" + args.attack + "/" + args.white_model + "/" + args.img_name
    # img_path_adv = "/home/shuxue2/yy/TransferAttackv2/adv_data/admix/resnet18/ILSVRC2012_val_00000020.JPEG"
    rgb_img_adv = cv2.imread(img_path_adv, 1)[:, :, ::-1]
    rgb_img_adv = np.float32(rgb_img_adv) / 255
    input_tensor_adv = preprocess_image(rgb_img_adv,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(args.device)
    # read raw_image
    img_path_raw = "/home/shuxue2/yy/TransferAttackv2/new_data/images/" + args.img_name
    rgb_img_raw = cv2.imread(img_path_raw, 1)[:, :, ::-1]
    rgb_img_raw = np.float32(rgb_img_raw) / 255
    input_tensor_raw = preprocess_image(rgb_img_raw,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).to(args.device)

    out_logit = model(input_tensor_raw)
    # print(out_logit.shape)  #torch.Size([1, 1000])
    # pretop2= torch.topk(out_logit, 2) #前k个元素
    # with open('imagenet_classes.txt') as f:
        # classes = [line.strip() for line in f.readlines()]

    # percentage = torch.nn.functional.softmax(out_logit, dim=1)[0] * 100
    # _, index = torch.max(out_logit, 1) # 最大值
    # print(classes[index[0]], percentage[index[0]].item())
    # _, indices = torch.sort(out_logit, descending=True)
    # print([(idx, percentage[idx].item()) for idx in indices[0][:5]]) # 查看前k个预测结果和置信度

    idx = np.argmax(out_logit.cpu().data.numpy())
    # print(idx)

    index = find_index(target_name = args.img_name)
    if idx != index:
        print(f'{args.img_name} prediction wrong, true label is {index}, but now it is {idx}.')
    else:
        print(f"Prediction {index} is right, can continue!")
    # 如果想要若原来就判错的图像不进行后续操作，在此后所有语句缩进到else块中。
    # 但可以发现，即便是原本判错的，对抗图像会以更高的置信度让机器错判(例如，00000051)。

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputTarget(281)]
    targets = None  ##设置类别

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                        target_layers=target_layers) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        #adv
        grayscale_cam_adv = cam(input_tensor=input_tensor_adv,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        grayscale_cam_adv = grayscale_cam_adv[0, :]
        heatmap_adv, cam_image_adv = show_cam_on_image(rgb_img_adv, grayscale_cam_adv, use_rgb=True, image_weight=0.5)
        heatmap_adv = cv2.cvtColor(heatmap_adv, cv2.COLOR_RGB2BGR)
        cam_image_adv = cv2.cvtColor(cam_image_adv, cv2.COLOR_RGB2BGR)
        #raw
        grayscale_cam_raw = cam(input_tensor=input_tensor_raw,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

        grayscale_cam_raw = grayscale_cam_raw[0, :]
        heatmap_raw, cam_image_raw = show_cam_on_image(rgb_img_raw, grayscale_cam_raw, use_rgb=True, image_weight=0.5)
        heatmap_raw = cv2.cvtColor(heatmap_raw, cv2.COLOR_RGB2BGR)
        cam_image_raw = cv2.cvtColor(cam_image_raw, cv2.COLOR_RGB2BGR)
    # adv
    uniform_heatmap_adv = make_uniform_heatmap(heatmap_adv, square=7)
    gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
    gb_adv = gb_model(input_tensor_adv, target_category=None)

    cam_mask_adv = cv2.merge([grayscale_cam_adv, grayscale_cam_adv, grayscale_cam_adv])
    cam_gb_adv = deprocess_image(cam_mask_adv * gb_adv)
    gb_adv = deprocess_image(gb_adv)
    # raw
    #可以调整成224
    heatmap_raw = cv2.resize(heatmap_raw, (224, 224))
    uniform_heatmap_raw = make_uniform_heatmap(heatmap_raw, square=7)
    gb_raw = gb_model(input_tensor_raw, target_category=None)

    cam_mask_raw = cv2.merge([grayscale_cam_raw, grayscale_cam_raw, grayscale_cam_raw])
    cam_gb_raw = deprocess_image(cam_mask_raw * gb_raw)
    gb_raw = deprocess_image(gb_raw)

# Save adv images
    name, ext = os.path.splitext(args.img_name)
    output_dir_adv = args.output_dir + '/' + args.target_model + '/' +  name + '/adv'
    os.makedirs(output_dir_adv, exist_ok=True)

    camunf_adv_output_path = os.path.join(output_dir_adv, f'{args.method}_camunf.jpg')
    hm_adv_output_path = os.path.join(output_dir_adv, f'{args.method}_hm.jpg')
    cam_adv_output_path = os.path.join(output_dir_adv, f'{args.method}_cam.jpg')
    gb_adv_output_path = os.path.join(output_dir_adv, f'{args.method}_gb.jpg')
    cam_gb_adv_output_path = os.path.join(output_dir_adv, f'{args.method}_cam_gb.jpg')
    cv2.imwrite(camunf_adv_output_path, uniform_heatmap_adv)
    cv2.imwrite(hm_adv_output_path, heatmap_adv)
    cv2.imwrite(cam_adv_output_path, cam_image_adv)
    cv2.imwrite(gb_adv_output_path, gb_adv)  # 仅关注对输出起到正向贡献的且激活了的特征，能够减少显著图噪音,相当于可视化正向梯度
    cv2.imwrite(cam_gb_adv_output_path, cam_gb_adv)

    # Save adv images
    output_dir_raw = args.output_dir + '/' + args.target_model + '/' + name + '/raw'
    os.makedirs(output_dir_raw, exist_ok=True)

    camunf_raw_output_path = os.path.join(output_dir_raw, f'{args.method}_camunf.jpg')
    hm_raw_output_path = os.path.join(output_dir_raw, f'{args.method}_hm.jpg')
    cam_raw_output_path = os.path.join(output_dir_raw, f'{args.method}_cam.jpg')
    gb_raw_output_path = os.path.join(output_dir_raw, f'{args.method}_gb.jpg')
    cam_gb_raw_output_path = os.path.join(output_dir_raw, f'{args.method}_cam_gb.jpg')
    cv2.imwrite(camunf_raw_output_path, uniform_heatmap_raw)
    cv2.imwrite(hm_raw_output_path, heatmap_raw)
    cv2.imwrite(cam_raw_output_path, cam_image_raw)
    cv2.imwrite(gb_raw_output_path, gb_raw)  # 仅关注对输出起到正向贡献的且激活了的特征，能够减少显著图噪音,相当于可视化正向梯度
    cv2.imwrite(cam_gb_raw_output_path, cam_gb_raw)
    print("Saving Done!", args.img_name)
    # Note: 先输出adv的类别和置信度

    # image = cv2.imread('/home/shuxue2/yy/TransferAttackv2/pytorch-grad-cam-master/output/ILSVRC2012_val_00000051/raw/gradcam_cam.jpg')
    # img = Image.fromarray(heatmap_raw)
    # print(img.mode)





