import os
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-arch", type=str, default="A")
parser.add_argument("-ds", type=int, default=4)
parser.add_argument("-dt", type=int, default=15)
parser.add_argument("-T", type=int, default=60)
parser.add_argument("-seed", type=int, default=118)
parser.add_argument("-epoch", type=int, default=0)
parser.add_argument("-num", type=int, default=3)
parser.add_argument("-num_epochs", type=int, default=100)
parser.add_argument("-batch_size_train", type=int, default=32)
parser.add_argument("-batch_size_test", type=int, default=4)

# save
parser.add_argument("-name", type=str, default=None)
parser.add_argument(
    "-modelPath",
    type=str,
    default=os.path.dirname(os.path.abspath(__file__)) + os.sep + "result/",
)
parser.add_argument("-modelNames", type=str, default=None)
parser.add_argument("-recordPath", type=str, default=None)
parser.add_argument("-recordNames", type=str, default=None)

# Data
parser.add_argument("-fbs", type=str, default=False)
parser.add_argument("-lasso_lambda", type=float, default=1e-8)
parser.add_argument("-sparsity_ratio", type=float, default=0.5)
parser.add_argument("-dataset", type=str, default="gesture")
parser.add_argument("-data_path", type=str, default="/data1/DVSGesture")

# FBS
parser.add_argument("-clip", type=int, default=10)
parser.add_argument("-is_train_Enhanced", type=bool, default=True)
parser.add_argument("-is_spike", type=bool, default=False)
parser.add_argument("-interval_scaling", type=bool, default=False)

# Gate
parser.add_argument("-gate", type=bool, default=False)
parser.add_argument("-target_rate", type=float, default=0.7)

# Param
parser.add_argument("-init_method", type=str, default=None)
parser.add_argument("-pretrained_path", default=None)
parser.add_argument("-criterion", default=nn.MSELoss())

# Network
parser.add_argument("-in_channels", default=2)
parser.add_argument("-num_classes", type=int, default=11)
parser.add_argument("-beta", type=float, default=0)
parser.add_argument("-alpha", type=float, default=0.3)
parser.add_argument("-Vreset", type=float, default=0)
parser.add_argument("-Vthres", type=float, default=0.3)
parser.add_argument("-mem_act", default=torch.relu)
parser.add_argument("-mode_select", type=str, default="spike")
parser.add_argument("-TR_model", type=str, default="NTR")
parser.add_argument("-track_running_stats", type=bool, default=True)
parser.add_argument("-lens", type=float, default=0.25)
parser.add_argument("-ps", type=str)

# optimizer
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-betas", default=[0.9, 0.999])
parser.add_argument("-eps", type=float, default=1e-8)
parser.add_argument("-weight_decay", type=float, default=1e-4)
parser.add_argument("-lr_scheduler", type=bool, default=True)

# Attention
parser.add_argument("-attention", type=str, default="no")
parser.add_argument("-attention_before_conv", type=bool, default=False)
parser.add_argument("-attention_per_time", type=bool, default=False)
parser.add_argument("-attention_in_spike", type=bool, default=False)
parser.add_argument("-reduction", type=int, default=1)
parser.add_argument("-lam", type=float, default=0.5)

# Dataloder
parser.add_argument("-drop_last", type=bool, default=False)
parser.add_argument("-pip_memory", type=bool, default=True)
parser.add_argument("-num_work", type=int, default=4)

parser.add_argument("-collect", type=str, default="firing")
parser.add_argument("-feature_name", type=str, default="S_0")
