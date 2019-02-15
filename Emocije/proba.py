# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:06:37 2018

@author: pc2012
"""

from scipy.io import arff
import pandas as pd

data = arff.loadarff('scene.arff')
df = pd.DataFrame(data[0])
mapping = {b'0': 0, b'1': 1}
df.replace(mapping, inplace = True)
