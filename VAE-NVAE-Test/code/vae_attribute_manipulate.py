# -*- coding: utf-8 -*-
import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import cv2
from PIL import Image
import pandas as pd

from nvae.dataset import ImageAttrDataset
from nvae.utils import add_sn
from nvae.vae_celeba import NVAE
from nvae.utils import reparameterize



#用于分离属性向量
def compute_attribute_vector(model, image_path, image_size, attrs, data_attrs, male_attribute_vectors_file, female_attribute_vectors_file, device):

    pos_male_vectors = torch.zeros(len(attrs), 1, 512, 2, 2)
    pos_female_vectors = torch.zeros(len(attrs), 1, 512, 2, 2)
    neg_male_vectors = torch.zeros(len(attrs), 1, 512, 2, 2)
    neg_female_vectors = torch.zeros(len(attrs), 1, 512, 2, 2)

    pos_male_nums = torch.zeros(len(attrs), 1)
    pos_female_nums = torch.zeros(len(attrs), 1)
    neg_male_nums = torch.zeros(len(attrs), 1)
    neg_female_nums = torch.zeros(len(attrs), 1)

    # train/0
    dataset_path = image_path   #image_path, '0'), os.path.join(image_path, '1')]
    # dataset_path = [opt.dataset_path]

    train_ds = ImageAttrDataset(dataset_path, img_dim=image_size, attrs=data_attrs)
    #print(len(train_ds))
    print('dataset_num:' + str(len(train_ds)))
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=4)
    count = 0
    #print('---------------------------------------------------------------------------')
    for image,labels in train_dataloader:
        #image, labels = data
        #print(image)
        count += 1
        #print(labels)
        #print(labels['Blond_Hair'])
        image = image.to(device)
        #print(labels['Male'])
        #print(image)

        mu, log_var, xs = model.encoder(image)
        z = reparameterize(mu, torch.exp(0.5 * log_var))
        z = z.detach().cpu()

        for i, attr in enumerate(attrs):
            #print('--------------------123--------------------------')
            #print(attr)
            #print(i)
            if labels[attr] == 1:
                if labels['Male'] == 1:
                    pos_male_vectors[i] += z[0]
                    pos_male_nums[i] += 1
                else:
                    pos_female_vectors[i] += z[0]
                    pos_female_nums[i] += 1
            else:
                #print('-----------------------')
                if labels['Male'] == 1:
                    #print('Male++++++++!!Yeah')
                    neg_male_vectors[i] += z[0]
                    neg_male_nums[i] += 1
                    #print('得分为：！！！！！！！！！！！！！！')
                    #print(neg_male_nums[i])
                    #print('Okkkkkkkkkkkkkkkkkkkkkkkkk!')
                else:
                    #print('Female++++++++++!!')
                    neg_female_vectors[i] += z[0]
                    #print('女性得分为：！！！！！！！！！！！！！！')
                    neg_female_nums[i] += 1
                    #print(neg_female_nums[i])
                    #print('对了这就！！！！')
        print(count)
    for i, num in enumerate(pos_male_nums):
            pos_male_vectors[i] /= num

    for i, num in enumerate(pos_female_nums):
            pos_female_vectors[i] /= num

    for i, num in enumerate(neg_male_nums):
            neg_male_vectors[i] /= num

    for i, num in enumerate(neg_female_nums):
            neg_female_vectors[i] /= num

    # print(pos_nums)
    # print(pos_vectors.shape)
    # print(pos_vectors)
    with torch.no_grad():
        male_attribute_vectors = {}
        female_attribute_vectors = {}

        #测试图片
        # for i in range(len(attrs)):
        #     pos_female_images = model.decoder(pos_female_vectors[i].to(device))
        #     neg_female_images = model.decoder(neg_female_vectors[i].to(device))
        #
        #     pos_male_images = model.decoder(pos_male_vectors[i].to(device))
        #     neg_male_images = model.decoder(neg_male_vectors[i].to(device))
        #
        #     plot_image([img_renorm(pos_female_images[0][0].permute(1, 2, 0).cpu())],
        #                [img_renorm(neg_female_images[0][0].permute(1, 2, 0).cpu())],
        #                'female'+attrs[i])
        #
        #     plot_image([img_renorm(pos_male_images[0][0].permute(1, 2, 0).cpu())],
        #                [img_renorm(neg_male_images[0][0].permute(1, 2, 0).cpu())],
        #                'male'+attrs[i])

        for i in range(len(attrs)):
            male_attribute_vectors[attrs[i]] = pos_male_vectors[i].cpu() - neg_male_vectors[i].cpu()
            female_attribute_vectors[attrs[i]] = pos_female_vectors[i].cpu() - neg_female_vectors[i].cpu()
            # draw the attribute for debugging
            print(attrs[i])
        
        torch.save(male_attribute_vectors, male_attribute_vectors_file)
        torch.save(female_attribute_vectors, female_attribute_vectors_file)
        # print(male_attribute_vectors)
        # print(female_attribute_vectors)
        return male_attribute_vectors, female_attribute_vectors


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    image = torch.unsqueeze(image, dim=0)
    return image


def save_image(model, folder, image_file_name, z):
    if not os.path.exists(folder):
        os.mkdir(folder)

    gen_img, _ = model.decoder(z)
    # print(gen_img.shape)
    gen_img = gen_img.permute(0, 2, 3, 1)
    gen_img = gen_img[0].cpu().numpy() * 255
    img = Image.fromarray(np.uint8(gen_img))

    img.save(os.path.join(folder, image_file_name))


#向图像中添加属性
def trans_attributes(model, image_path, save_path, male_attribute_vectors, female_attribute_vectors, male_attribute_vectors_file, female_attribute_vectors_file, attrs, device):
    male_attribute_vectors = torch.load(male_attribute_vectors_file)
    female_attribute_vectors = torch.load(female_attribute_vectors_file)

    model.eval()
    flag = -1
    dataset_path = image_path
    with torch.no_grad():
        for i in range(1,3):
            flag += 1
            #保存图片的路径也就是
            target_path = save_path
            #for root, dirs, files in os.walk(image_files):   image_files就是dataset_path，即读入的图片的路径
            cnt = 0
            if flag == 0:#对女性进行处理
                dataset_path = 'data/celeba/10client/2/'
                target_path = os.path.join(target_path, '0')
                for f in os.listdir(dataset_path):  #这里的f是每一个图片文件,遍历该文件夹下的所有图片文件
                    image = read_image(os.path.join(dataset_path, f)) #read_image函数来读取每一张图片
                    image = image.to(device)
                    mu, log_var, xs = model.encoder(image)
                    z = reparameterize(mu, torch.exp(0.5 * log_var))

                    z_r = z.detach()
                    for attr in attrs:
                        beta = -1
                        if flag == 0:
                            z_r += beta * female_attribute_vectors[attr].to(device)
                        else:
                            z_r += beta *  male_attribute_vectors[attr].to(device)

                    #这里的图片名称已经是f了，也就是文件夹中遍历的每一个文件的名称了
                    save_image(model, folder=target_path, image_file_name=f, z=z_r)

                    cnt += 1

                print(target_path, cnt)

            else:
                dataset_path = 'data/celeba/10client/1/' #这里读取的是男性图片的文件
                target_path = os.path.join(target_path, '1')
                for f in os.listdir(dataset_path):  # 这里的f是每一个图片文件,遍历该文件夹下的所有图片文件
                    image = read_image(os.path.join(dataset_path, f))  # read_image函数来读取每一张图片
                    image = image.to(device)
                    mu, log_var, xs = model.encoder(image)
                    z = reparameterize(mu, torch.exp(0.5 * log_var))

                    z_r = z.detach()
                    for attr in attrs:
                        beta = -1
                        if flag == 0:
                            z_r += beta * female_attribute_vectors[attr].to(device)
                        else:
                            z_r += beta * male_attribute_vectors[attr].to(device)

                    # 这里的图片名称已经是f了，也就是文件夹中遍历的每一个文件的名称了

                    save_image(model, folder=target_path, image_file_name=f, z=z_r)

                    cnt += 1

                print(target_path, cnt)


#主函数
def main():
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser()

    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--image_path", type=str, default='data/celeba/10client/0/')
    parser.add_argument('--img_sz', type=int, default=64, help='image size')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--num_attrs', type=int, default=1)
    parser.add_argument("--attribute_vectors", type=str, default='checkpoints/attribute/')
    parser.add_argument('--save_path', type=str, default='data/celeba/newpic/')

    opt = parser.parse_args()

    data_attrs, attrs, attribute_vectors = None, None, None
    # 'Blond_Hair', 'Narrow_Eyes', 'Smiling', 'Straight_Hair'
    if opt.num_attrs == 1:
        data_attrs = ['Smiling', 'Male']
        attrs = ['Smiling']
        attribute_vectors = os.path.join(opt.attribute_vectors, 'VAE-f1-'+attrs[0])
        opt.save_path = os.path.join(opt.save_path, 'VAE-f1-' + attrs[0])

    elif opt.num_attrs == 2:
        data_attrs = ['Blond_Hair', 'Narrow_Eyes', 'Male']
        attrs = ['Blond_Hair', 'Narrow_Eyes']
        attribute_vectors = os.path.join(opt.attribute_vectors, 'VAE-f2-'+attrs[0]+'-' +attrs[1])
        opt.save_path = os.path.join(opt.save_path, 'VAE-f2-'+attrs[0]+'-'+attrs[1])

    elif opt.num_attrs == 3:
        data_attrs = ['Blond_Hair', 'Narrow_Eyes', 'Smiling', 'Male']
        attrs = ['Blond_Hair', 'Narrow_Eyes', 'Smiling']
        attribute_vectors = os.path.join(opt.attribute_vectors, 'VAE-f3-' + attrs[0] + '-' + attrs[1] + '-' + attrs[2])
        opt.save_path = os.path.join(opt.save_path, 'VAE-f3-' + attrs[0] + '-' + attrs[1] + '-' + attrs[2])

    elif opt.num_attrs == 4:
        data_attrs = ['Blond_Hair', 'Narrow_Eyes', 'Smiling', 'Straight_Hair', 'Male']
        attrs = ['Blond_Hair', 'Narrow_Eyes', 'Smiling', 'Straight_Hair']
        attribute_vectors = os.path.join(opt.attribute_vectors, 'VAE-f4')
        opt.save_path = os.path.join(opt.save_path, 'VAE-f4')


    if not os.path.exists(attribute_vectors):
        os.mkdir(attribute_vectors)
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)


    device = "cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu"

    #pretrained_weights = ['470_0.639811.pth', '452_0.634575.pth', '400_0.692572.pth', '475_0.705544.pth', '400_0.660960.pth', '409_0.626322.pth', '475_0.694686.pth', '403_0.642286.pth', '425_0.674690.pth', '475_0.636303.pth']
    pretrained_weights = '347_0.766766.pth'

    male_attribute_vectors, female_attribute_vectors = None, None
    #for i in range(opt.clients):

    model = NVAE(z_dim=512, img_dim=(opt.img_sz, opt.img_sz))

    # apply Spectral Normalization
    model.apply(add_sn)
    model.to(device)
    state_dict = os.path.join('checkpoints/celeba0/', pretrained_weights)
    model.load_state_dict(torch.load(state_dict, map_location=device), strict=False)
    model.eval()


    print('-------client{}-------'.format(str(0)))
    image_path = opt.image_path
    save_path = opt.save_path

    print('image path: ' + image_path)
    print('state dict: ' + state_dict)
    print('attribute vectors: ' + attribute_vectors)
    print('save path: ' + save_path)

    male_attribute_vectors_file = os.path.join(attribute_vectors, 'male_attribute_vectors_'+str(0)+'.t')
    female_attribute_vectors_file = os.path.join(attribute_vectors, 'female_attribute_vectors_' + str(0) + '.t')

    male_attribute_vectors, female_attribute_vectors = compute_attribute_vector(model, image_path, opt.img_sz, attrs, data_attrs, male_attribute_vectors_file, female_attribute_vectors_file, device)
    trans_attributes(model, image_path, save_path, male_attribute_vectors, female_attribute_vectors, male_attribute_vectors_file, female_attribute_vectors_file, attrs, device)


if __name__ == '__main__':
    main()
