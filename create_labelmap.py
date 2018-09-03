import argparse

parser = argparse.ArgumentParser(description='Generate labelmap txt file')
parser.add_argument('--model', help="model name", required=True)

args = parser.parse_args()
item = 'item {{\n  name: {lab}\n  id: {num}\n}}\n\n'
with open("classes_{}.txt".format(args.model),"rb") as cls:
    lines = [l.strip() for l in cls.readlines()]
    with open("labelmap_{}.pbtxt".format(args.model),"w") as labmap:
        for ind, label in enumerate(lines):
            if len(label) > 0:
                if label[0] != "'":
                    label = "'" + label
                if label[len(label) - 1] != "'":
                    label = label + "'"
                labmap.writelines(item.format(lab = label,num = ind + 1))

