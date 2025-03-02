import argparse
import os
from model.utils.parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from model.data_loader.loader import ContentData, Random_StyleKoreanDataset2
from model.models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from model.models.diffusion import Diffusion
import torchvision
import torch.distributed as dist
from model.utils.util import fix_seed
import shutil
from PIL import Image

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """ set mulit-gpu """
    # dist.init_process_group(backend='nccl')
    # local_rank = dist.get_rank()
    # torch.cuda.set_device(local_rank)

    load_content = ContentData()

    output_path = opt.output_path

    SPACE = '쒨'

    temp_texts = []
    for ch in opt.input_sentence:
        if '가' <= ch <= '힣':
            temp_texts.append(ch)
        elif ch == ' ':
            temp_texts.append(SPACE)
    print(f'입력: {"".join(temp_texts).replace(SPACE, " ")}')

    style_dataset = Random_StyleKoreanDataset2(opt.style_path, opt.style_path, len(temp_texts))


    print('this process handle characters: ', len(style_dataset))
    style_loader = torch.utils.data.DataLoader(style_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                                pin_memory=True
                                                )
    

    diffusion = Diffusion(device=opt.device)

    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM).to(opt.device)
    
    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0: 
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location=torch.device('cpu')))
        print('load pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('input the correct checkpoint path')
    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae = vae.to(opt.device)
    # Freeze vae and text_encoder
    vae.requires_grad_(False)


    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)


    """generate the handwriting datasets"""
    loader_iter = iter(style_loader)

    for idx, x_text in tqdm(enumerate(temp_texts), position=0, desc='batch_number'):
        if x_text == SPACE:
            width, height = 64, 128
            empty_image = Image.new("RGB", (width, height), (255, 255, 255))
            empty_image.save(os.path.join(output_path, f'{str(idx)}_ .png'))
            continue

        data = next(loader_iter)
        data_val, laplace = data['style'][0], data['laplace'][0]
        
        data_loader = []

        # split the data into two parts when the length of data is two large
        if len(data_val) > 224:
            data_loader.append((data_val[:224], laplace[:224]))
            data_loader.append((data_val[224:], laplace[224:]))
        else:
            data_loader.append((data_val, laplace))
            
        for (data_val, laplace) in data_loader:
            style_input = data_val.to(opt.device)
            laplace = laplace.to(opt.device)
            text_ref = load_content.get_content(x_text)
            text_ref = text_ref.to(opt.device).repeat(style_input.shape[0], 1, 1, 1)
            # x = torch.randn((text_ref.shape[0], 4, style_input.shape[2]//8, (text_ref.shape[1]*32)//8)).to(opt.device)
            x = torch.randn((text_ref.shape[0], 4, 16, 16), device=opt.device)
            if opt.sample_method == 'ddim':
                ema_sampled_images = diffusion.ddim_sample(unet, vae, style_input.shape[0], 
                                                        x, style_input, laplace, text_ref,
                                                        opt.sampling_timesteps, opt.eta)
            elif opt.sample_method == 'ddpm':
                ema_sampled_images = diffusion.ddpm_sample(unet, vae, style_input.shape[0], 
                                                        x, style_input, laplace, text_ref)
            else:
                raise ValueError('sample method is not supported')
            
            for index in range(len(ema_sampled_images)):
                im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                image = im.convert("L")
                image.save(os.path.join(output_path, f'{str(idx)}_{x_text}.png'))
        
    image_files = sorted([f for f in os.listdir(output_path) if f.endswith(".png")], key=lambda s: int(s.split('_')[0]))
    images = [Image.open(os.path.join(output_path, img)) for img in image_files]

    height = images[0].size[1]
    total_width = sum(img.size[0] for img in images)
    result = Image.new("RGB", (total_width, height))

    x_offset = 0
    for img in images:
        result.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    result.save(os.path.join(output_path, f'{opt.input_sentence}.png'))

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='model/configs/Korean_scratch.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--one_dm', dest='one_dm', default='model/ckpt/final_model.pt', required=True, help='pre-train model for generating')
    parser.add_argument('--device', type=str, default='cuda', help='device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    parser.add_argument('--input_sentence', type=str, default='입력 문장 기본값', help='just input')
    parser.add_argument('--style_path', type=str, default='style.png', help='style image path')
    parser.add_argument('--output_path', type=str, default='output', help='output path for generated images')

    opt = parser.parse_args()
    main(opt)