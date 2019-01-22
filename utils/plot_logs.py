import os, subprocess, json, ast, sys, re, random, csv, glob
from utils import elastic as es
from config import config
from utils import utf8_helper
dirname = os.path.dirname(__file__)

cnf = config.Config()

log_path = os.path.join(dirname, '../data/runs/validation/')

os.chdir(log_path)
for file in glob.glob("*.log"):
    plot_title = file.split(".")[0]
    print(plot_title)

# for in