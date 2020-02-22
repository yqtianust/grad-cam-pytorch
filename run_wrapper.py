from utils import create_folder
import os
import subprocess

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    img_dirs = ['./img/org', './img/bg_0', './img/bg_127', './img/bg_255', './img/obj_0', './img/obj_127',
                './img/obj_255']

    arch = "resnet152"
    target_layer = "layer4"
    cuda = True
    topk = 1
    for img_dir in img_dirs:

        output_dir = "./cam_{}_{}_{}".format(arch, target_layer, img_dir.replace("/", '_'))
        create_folder(output_dir)
        image_paths = []

        for i in range(0, 50000):
            filename = "ILSVRC2012_val_000{:05}.JPEG".format(i + 1)
            image_path = os.path.join(img_dir, filename)
            image_paths.append(image_path)

        for images in list(chunks(image_paths, n=16)):
            # process_a_batch(images, target_layer, arch, topk, output_dir, cuda)
            arguments = ["python3","run.py", "-a", "{}".format(arch),
                         "-t", "{}".format(target_layer),
                         "-o", "{}".format(output_dir), "-i"]
            for img in images:
                # arguments.append("-i")
                arguments.append("{}".format(img))
            print(arguments)
            subprocess.call(arguments)

if __name__ == '__main__':
    main()