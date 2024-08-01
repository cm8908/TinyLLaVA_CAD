import sys; sys.path.append(".."); sys.path.append(".")
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d
from PIL import Image
from OCC.Extend.DataExchange import read_step_file
from OCC.Display.SimpleGui import init_display
from pythonocc_operator.lib.extrude import CADSequence
from pythonocc_operator.lib.visualize import create_CAD

def show(path):
    out_shape = read_step_file(path)
    display, start_display, _, _ = init_display()
    display.DisplayShape(out_shape, update=True)
    start_display()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default=None)
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--pc', type=str, default=None)
    args = parser.parse_args()

    if args.img:
        img = Image.open(args.img)
        plt.imshow(np.array(img))
        plt.show()
    
    if args.step:
        if os.path.isdir(args.step):
            raise NotImplementedError
            for f in os.listdir(args.step):
                if f.endswith('.step'):
                    show(os.path.join(args.step, f))
        else:
            show(args.step)
    
    if args.pc:
        pc = o3d.io.read_point_cloud(args.pc)
        o3d.visualization.draw_geometries([pc])

if __name__ == "__main__":
    main() 