import argparse
import cv2
import csv
import numpy as np
import torch
import timm
import os

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image, make_uniform_heatmap, deprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default='cuda:2',
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--img-name', type=str, default='ILSVRC2012_val_00000051.JPEG', help='Input image name')
    parser.add_argument('--white-model', type=str, default='resnet18', help='White model')
    parser.add_argument('--attack', type=str, default='admix', help='Attack method')
    parser.add_argument('--target-model', type=str, default='vit_base_patch16_224',
                        help='Target model:vit_base_patch16_224, pit_b_224,visformer_small, swin_tiny_patch4_window7_224')

    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory to save the images')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def find_index(target_name, file_path='/home/shuxue2/yy/TransferAttackv2/new_data/labels.csv'):
    # 请替换为你的CSV文件路径
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if row[0] == target_name:
                return int(row[1])  # 返回第二列的序号，转换为整数类型
    return -1


def reshape_transform_swin(tensor, height=7, width=7):  # target_model
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform_vit_pit(tensor, height=14, width=14):  # target_model
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



if __name__ == '__main__':
    """ python vit_cam.py --image-path <./examples/both.png>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    filepath = timm.create_model(args.target_model).pretrained_cfg['hf_hub_id']
    pthname = os.path.basename(filepath)
    pathname = '/home/shuxue2/.cache/huggingface/hub/models--timm--' + pthname + '/pytorch_model.bin'
    model = timm.create_model(args.target_model, pretrained=True, pretrained_cfg_overlay=dict(file=pathname))
    # model = torch.hub.load('facebookresearch/deit:main',
    # 'deit_tiny_patch16_224', pretrained=True)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    # for n, v in model.named_parameters(): #获取模块名称
    #     print(n)

    # ViT: model.blocks[-1].norm1
    # PiT: model.transformers[-1].blocks[-1].norm1
    # visformer: model.stage3[-1].norm1
    # Swin: model.layers[-1].blocks[-1].norm2
    if 'visformer' in args.target_model:
        target_layers = [model.stage3[-1].norm1]
    elif 'swin' in args.target_model:
        target_layers = [model.layers[-1].blocks[-1].norm2]
    elif 'pit' in args.target_model:
        target_layers = [model.transformers[-1].blocks[-1].norm1]
    else:
        target_layers = [model.blocks[-1].norm1]
    # target_layers = [model.transformers[-1].blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    # ViT：14; PiT:8; Swin:7(reshape前不取通道); Visformer: None(本身就是仿照CNN结构改的无需改变维度)

    def reshape_transform(tensor):
        if 'visformer' in args.target_model:
            return None  # Visformer: None(本身就是仿照CNN结构改的无需改变维度)
        elif 'swin' in args.target_model:
            return reshape_transform_swin(tensor)
        elif 'pit' in args.target_model:
            return reshape_transform_vit_pit(tensor, height=8, width=8)
        else:
            return reshape_transform_vit_pit(tensor, height=14, width=14)

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   # use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   # use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)
    # load images
    img_path_adv = "/home/shuxue2/yy/TransferAttackv2/adv_data/" + args.attack + "/" + args.white_model + "/" + args.img_name
    # img_path_adv = "/home/shuxue2/yy/TransferAttackv2/adv_data/admix/resnet18/ILSVRC2012_val_00000020.JPEG"
    rgb_img_adv = cv2.imread(img_path_adv, 1)[:, :, ::-1]
    rgb_img_adv = cv2.resize(rgb_img_adv, (224, 224))
    rgb_img_adv = np.float32(rgb_img_adv) / 255
    input_tensor_adv = preprocess_image(rgb_img_adv,
                                        mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
    # read raw_image
    img_path_raw = "/home/shuxue2/yy/TransferAttackv2/new_data/images/" + args.img_name
    rgb_img_raw = cv2.imread(img_path_raw, 1)[:, :, ::-1]
    rgb_img_raw = cv2.resize(rgb_img_raw, (224, 224))
    rgb_img_raw = np.float32(rgb_img_raw) / 255
    input_tensor_raw = preprocess_image(rgb_img_raw,
                                        mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])  # .to(args.use_cuda)
    # 测试是否原本就是预测正确的
    out_logit = model(input_tensor_raw.float().cuda())
    idx = np.argmax(out_logit.cpu().data.numpy())
    index = find_index(target_name=args.img_name)
    if idx != index:
        print(f'{args.img_name} prediction wrong, true label is {index}, but now it is {idx}.')
    else:
        print(f"Prediction {index} is right, can continue!")

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32
    # adv
    grayscale_cam_adv = cam(input_tensor=input_tensor_adv,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam_adv = grayscale_cam_adv[0, :]
    heatmap_adv, cam_image_adv = show_cam_on_image(rgb_img_adv, grayscale_cam_adv, image_weight=0.5)

    # raw
    grayscale_cam_raw = cam(input_tensor=input_tensor_raw,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)
    grayscale_cam_raw = grayscale_cam_raw[0, :]
    heatmap_raw, cam_image_raw = show_cam_on_image(rgb_img_raw, grayscale_cam_raw, image_weight=0.5)

    if 'visformer' in args.target_model:  # 此时需要转换通道保持热图正确
        heatmap_adv = cv2.cvtColor(heatmap_adv, cv2.COLOR_RGB2BGR)
        cam_image_adv = cv2.cvtColor(cam_image_adv, cv2.COLOR_RGB2BGR)
        heatmap_raw = cv2.cvtColor(heatmap_raw, cv2.COLOR_RGB2BGR)
        cam_image_raw = cv2.cvtColor(cam_image_raw, cv2.COLOR_RGB2BGR)

    # adv
    uniform_heatmap_adv = make_uniform_heatmap(heatmap_adv, square=7)
    gb_model = GuidedBackpropReLUModel(model=model, device=args.use_cuda)
    gb_adv = gb_model(input_tensor_adv, target_category=None)

    cam_mask_adv = cv2.merge([grayscale_cam_adv, grayscale_cam_adv, grayscale_cam_adv])
    cam_gb_adv = deprocess_image(cam_mask_adv * gb_adv)
    gb_adv = deprocess_image(gb_adv)
    # raw
    # 可以调整成224
    heatmap_raw = cv2.resize(heatmap_raw, (224, 224))
    uniform_heatmap_raw = make_uniform_heatmap(heatmap_raw, square=7)
    gb_raw = gb_model(input_tensor_raw, target_category=None)

    cam_mask_raw = cv2.merge([grayscale_cam_raw, grayscale_cam_raw, grayscale_cam_raw])
    cam_gb_raw = deprocess_image(cam_mask_raw * gb_raw)
    gb_raw = deprocess_image(gb_raw)

    # Save adv images
    name, ext = os.path.splitext(args.img_name)
    output_dir_adv = args.output_dir + '/' + args.target_model + '/' + name + '/adv'
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
