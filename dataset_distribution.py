from glob import glob
import wfdb
import numpy as np
from tqdm import tqdm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from pylab import savefig

def get_records():
    """ Get paths for data in data/mit/ directory """
    # Download if doesn't exist

    # There are 3 files for each record
    # *.atr is one of them
    paths = glob('./mit_arrythmia_dat/*.atr')

    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()

    return paths

def segmentation(records):

    dataset = []
    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels=[0])
        for s in tqdm(signals):
            dataset.append(s[0])
        break
    return dataset


if __name__ == "__main__":
    records = get_records()
    """'N' for normal beats. Similarly we can give the input 'L' for left bundle branch block beats. 'R' for right bundle branch block
        beats. 'A' for Atrial premature contraction. 'V' for premature ventricular contraction. '/' for paced beat. 'E' for Ventricular
        escape beat."""

    sgs = segmentation(records)
    df_cm = pd.DataFrame(sgs)

    svm = sn.heatmap(df_cm, annot=True, cmap='coolwarm', linecolor='white', linewidths=1)

    figure = svm.get_figure()
    figure.savefig('svm_conf.png', dpi=400)