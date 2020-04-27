import os
import argparse
import cv2
import numpy as np

# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument("hr_img_dir", help="path to high resolution image dir")
parser.add_argument("lr_img_dir", help="path to desired output dir for downsampled images")
parser.add_argument("-k", "--keepdims", help="keep original image dimensions in downsampled images",
                    action="store_true")
args = parser.parse_args(["hr_img_dir", "lr_img_dir"])

hr_image_dir = args.hr_img_dir
lr_image_dir = args.lr_img_dir

# create LR image dirs
os.makedirs(lr_image_dir + "/2x", exist_ok=True)
os.makedirs(lr_image_dir + "/3x", exist_ok=True)
os.makedirs(lr_image_dir + "/4x", exist_ok=True)
os.makedirs(lr_image_dir + "/6x", exist_ok=True)

supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")
# Downsample HR images
for filename in os.listdir(hr_image_dir):
    if not filename.endswith(supported_img_formats):
        continue

    # Read HR image
    hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])

    dst = np.empty_like(hr_img)  # create empty array the size of the image

    # noise = 0 standard deviation = sqrt(variance) where the given variance for the project is 0.01 ---> Therefore SD = 0.1
    noise = cv2.randn(dst, (0, 0, 0), (0.1, 0.1, 0.1))  # add random img noise

    # Pass img through noise filter to add noise
    img_noise = cv2.addWeighted(hr_img, 0.5, noise, 0.5, 50)

    # Blurring function; kernel=1.5, sigma=auto
    hr_img = cv2.GaussianBlur(img_noise, (int(1.5), int(1.5)), 0)

    # Blur with Gaussian kernel of width sigma = 1
    # hr_img = cv2.GaussianBlur(hr_img, (0,0), 1, 1)

    # Downsample image x2 with size set to 100*100
    lr_image_2x = cv2.resize(hr_img, (100, 100), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(lr_image_dir + "/2x", filename), lr_image_2x)

    # # Downsample image 3x
    # lr_img_3x = cv2.resize(hr_img, (0, 0), fx=(1 / 3), fy=(1 / 3),
    #                        interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_img_3x = cv2.resize(lr_img_3x, hr_img_dims,
    #                            interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(lr_image_dir + "/3x", filename), lr_img_3x)
    #
    # # Downsample image 4x
    # lr_img_4x = cv2.resize(hr_img, (0, 0), fx=0.25, fy=0.25,
    #                        interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
    #                            interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(lr_image_dir + "/4x", filename), lr_img_4x)
    #
    # # Downsample image 6x
    # lr_img_6x = cv2.resize(hr_img, (0, 0), fx=1 / 6, fy=1 / 6,
    #                        interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_img_4x = cv2.resize(lr_img_6x, hr_img_dims,
    #                            interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(lr_image_dir + "/6x", filename), lr_img_6x)
