# MGGPT
MGGPT framework for fraud detection in Cryptocurrency networks

Libraries Include:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
import os
from transformers import GPT2Model, GPT2Config
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
import os
