import os
import shutil
import numpy as np

negative_source = "./data/negative"
negative_dest = "./data/test/negative"

positive_source = "./data/positive"
postive_dest = "./data/test/positive"

def split(source, destination, ratio=0.2):
    files = os.listdir(source)
    for f in files:
        if np.random.rand(1) < ratio:
            shutil.move(source + '/'+ f, destination + '/'+ f)