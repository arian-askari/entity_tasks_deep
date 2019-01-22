import os, subprocess, json, ast, sys, re, random, csv, glob
from utils import elastic as es
from config import config
from utils import utf8_helper
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileMerger

dirname = os.path.dirname(__file__)
cnf = config.Config()

def pdf_merger(pdfs):
    # pdfs = sorted(pdfs, reverse=True)
    pdfs = sorted(pdfs)
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(open(pdf, 'rb'))

    with open('./plot/' +'result.pdf', 'wb') as fout:
        merger.write(fout)

# log_path = os.path.join(dirname, '../data/runs/validation/')
log_path = os.path.join(dirname, '../data/runs/validation/adam/')

os.chdir(log_path)
pdf_files = []
# files = sorted(glob.glob("*.log"))
files = glob.glob("*.log")
for file in files:

    plot_title = file.split(".")[0]
    n_count = re.findall(r'.*?L1N\((\d+)\)',plot_title)[0]
    e_count = re.findall(r'.*?epochCount\((\d+)\)', plot_title)[0]
    print(plot_title)

    # data = np.genfromtxt(file, delimiter=',', skip_header=10, skip_footer=10, names=['x', 'y', 'z'])
    data = np.genfromtxt(file, delimiter=',', skip_header=1, skip_footer=0, names=['epoch', 'acc', 'loss'])

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    plot_title = "One Layer - Neurons: " + str(n_count) + " Epoch: " + str(e_count)
    ax1.set_title(plot_title)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax1.set_xlim([0, 1000])
    ax1.set_ylim([0, 10])

    ax1.plot(data['epoch'], data['loss'], color='r', label='Loss Per Epoch')

    leg = ax1.legend()

    # plt.show()
    # pdf_path = './plot/' + plot_title + '.pdf'
    pdf_path = './plot/' + plot_title
    pdf_files.append(pdf_path + ".pdf")
    # plt.savefig(pdf_path, size=(300, 400), format='pdf', dpi=1000)
    # plt.savefig(pdf_path + ".png", size=(50, 150), format='png', dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path + ".pdf", size=(50, 150), format='pdf', dpi=300, bbox_inches='tight')


pdf_merger(pdf_files)

