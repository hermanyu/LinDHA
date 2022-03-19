# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:01:53 2022

@author: herman
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import PaulBettany as jarvis


train = pd.read_csv('../data/train.csv')
total = pd.read_csv('../data/ames-cleaned.csv')

proto = jarvis.Project(total[ : len(train.index) ],
                       total[ len(train.index) : ],
                       target='SalePrice',
                       name = 'prototype')



proto.prepare_data()

prescale1 = proto.X_train

prescale2 = proto.X_test

prescale2 = proto.X_unknown

prescale4 = proto.X

proto.scaler = StandardScaler()

proto.prepare_data()

postscale = proto.X_train

postscale1 = proto.X_train

postscale2 = proto.X_test

postscale2 = proto.X_unknown

postscale4 = proto.X