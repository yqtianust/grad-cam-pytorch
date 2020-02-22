import os
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import models
from main import get_device, get_classtable, save_gradcam, preprocess
from utils import create_folder
from grad_cam import (
    BackPropagation,
    GradCAM
)
import argparse


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

def save_cam(filename, gcam):
    # gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    cv2.imwrite(filename, np.uint8(cmap))
    del cmap


def load_images_only(image_paths):
    images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
    return images

# @click.group()
# @click.pass_context
# def main(ctx):
#     print("Mode:", ctx.invoked_subcommand)

# @main.command()
# @click.option("-i", "--image-paths", type=str, multiple=True, required=True)
# @click.option("-a", "--arch", type=click.Choice(model_names), required=True)
# @click.option("-t", "--target-layer", type=str, required=True)
# @click.option("-k", "--topk", type=int, default=3)
# @click.option("-o", "--output-dir", type=str, default="./results")
# @click.option("--cuda/--cpu", default=True)
def process_a_batch(image_paths, target_layer, arch, topk, output_dir, cuda):
    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model.to(device)
    model.eval()

    # Images
    images = load_images_only(image_paths)
    images = torch.stack(images).to(device)

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted

    # =========================================================================
    print("Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    for i in range(topk):
        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)
        regions = regions.cpu().numpy()
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(image_paths[j], classes[ids[j, i]], probs[j, i]))
            # Grad-CAM
            save_cam(
                filename=os.path.join(
                    output_dir,
                    "{}-{}-{}-{:.3}.png".format(
                        os.path.basename(image_paths[j]), classes[ids[j, i]], ids[j, i], probs[j, i]
                    ),
                ),
                gcam=regions[j, 0]
            )
    bp.remove_hook()
    gcam.remove_hook()
    del bp
    del images
    del gcam
    del model
    del regions
    del probs
    del ids
    del _
    torch.cuda.empty_cache()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main2():
    img_dirs = ['./img/org', './img/bg_0', './img/bg_127', './img/bg_255', './img/obj_0', './img/obj_127',
                './img/obj_255']

    arch = "resnet152"
    target_layer = "layer4"
    cuda = True
    topk = 1
    for img_dir in img_dirs[0:1]:

        output_dir = img_dir + "./cam_{}_{}_{}".format(arch, target_layer, img_dir.replace("/", '_'))
        create_folder(output_dir)
        image_paths = []

        for i in range(0, 50):
            filename = "ILSVRC2012_val_000{:05}.JPEG".format(i + 1)
            image_path = os.path.join(img_dir, filename)
            image_paths.append(image_path)

        for images in list(chunks(image_paths, n=16)):
            process_a_batch(images, target_layer, arch, topk, output_dir, cuda)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-paths", type=str, nargs ='*', required=True)
    parser.add_argument("-a", "--arch", type=str, required=True)
    parser.add_argument("-t", "--target-layer", type=str, required=True)
    # parser.add_argument("-k", "--topk", type=int, default=3)
    parser.add_argument("-o", "--output-dir", type=str, default="./results")
    # parser.add_argument("--cuda/--cpu", default=True)
    # process_a_batch()
    args = parser.parse_args()
    # print(args)
    process_a_batch(image_paths=args.image_paths, target_layer=args.target_layer,
                    arch = args.arch, output_dir=args.output_dir, topk=1, cuda=True)