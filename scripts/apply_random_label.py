from PIL import Image, ImageFont, ImageDraw
import os
import random
from tqdm import tqdm


def label_cycle(labels):
    while True:
        idx = random.randint(0, len(labels) - 1)
        yield idx, labels[idx]


def draw_label(img, label, fontsizes=(32, 64), colorrange=(122, 255)):
    draw = ImageDraw.Draw(img)

    fontsize = random.randint(fontsizes[0], fontsizes[1])

    while True:
        font = ImageFont.truetype('FreeMono.ttf', fontsize)
        textsize = draw.textsize(label, font)
        positions = (img.size[0] - textsize[0], img.size[1] - textsize[1])

        if positions[0] >= 0 and positions[1] >= 0:
            break
        else:
            fontsize -= 1
            assert fontsize > 0

    pos = (random.randint(0, positions[0]), random.randint(0, positions[1]))

    color = (random.randint(colorrange[0], colorrange[1]),
             random.randint(colorrange[0], colorrange[1]),
             random.randint(colorrange[0], colorrange[1]))

    draw.text(pos, label, color, font=font)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, required=True, help='Path to image data.')
    parser.add_argument('-l', '--labels', type=str, required=True, help='List of labels to draw on images.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output path to write altered images to. Directory tree from data will be preserved.')
    parser.add_argument('-f', '--fontsizes', type=int, nargs=2, required=False, default=[32, 64],
                        help='Min and max font size. Eventual size is chosen at random.')
    parser.add_argument('-c', '--color', type=int, nargs=2, required=False, default=[122, 255],
                        help='Min and max color value. Eventual color is chosen at random.')
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        labels = f.read().splitlines()

    print('Read %d labels.' % len(labels))

    assert os.path.exists(args.data)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    label_gen = label_cycle(labels)

    label_list = []
    category_index = 0

    nFiles = len(os.walk(args.data).next()[2])

    for path, dirs, files in tqdm(os.walk(args.data)):
        if len(files) == 0:
            continue

        dst = os.path.join(args.output, os.path.relpath(path, args.data))
        if not os.path.exists(dst):
            os.mkdir(dst)

        for file in files:
            img = Image.open(os.path.join(path, file))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            label_index, label = next(label_gen)
            draw_label(img, label, args.fontsizes, args.color)
            img.save(os.path.join(dst, file))

            label_list.append(
                '%s %d,%d' % (os.path.join(os.path.relpath(path, args.data), file), category_index, label_index))

        category_index += 1

    with open(os.path.join(args.output, 'caffe_style_list.txt'), 'w') as f:
        for label in label_list:
            f.write(label + '\n')
