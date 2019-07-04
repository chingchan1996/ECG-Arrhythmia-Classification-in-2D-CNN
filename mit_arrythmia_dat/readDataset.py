from glob import glob
import wfdb
import numpy as np
from tqdm import tqdm
import biosppy
import matplotlib.pyplot as plt
import cv2
import os
import math

def get_records():
    """ Get paths for data in data/mit/ directory """
    # Download if doesn't exist

    # There are 3 files for each record
    # *.atr is one of them
    paths = glob('./*.atr')

    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()

    return paths

def segmentation(records, type, output_dir=''):

    """'N' for normal beats. Similarly we can give the input 'L' for left bundle branch block beats. 'R' for right bundle branch block
        beats. 'A' for Atrial premature contraction. 'V' for ventricular premature contraction. '/' for paced beat. 'E' for Ventricular
        escape beat."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    kernel = np.ones((4, 4), np.uint8)
    count = 1

    '''
    max_values = []
    min_values = []
    mean_values = []
    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels=[0])
        mean_values.append(np.mean(signals))

    mean_v = np.mean(np.array(mean_values))
    std_v = 0
    count = 0
    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels=[0])
        count += len(signals)
        for s in signals:
            std_v += (s[0] - mean_v)**2

    std_v = np.sqrt(std_v/count)'''

    mean_v = -0.33859
    std_v = 0.472368
    floor = mean_v - 3*std_v
    ceil = mean_v + 3*std_v

    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels = [0])

        ann = wfdb.rdann(e, 'atr')
        good = [type]
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in tqdm(imp_beats):
            beats = list(beats)
            j = beats.index(i)
            if(j!=0 and j!=(len(beats)-1)):

                data = (signals[beats[j]-96: beats[j]+96, 0])

                results.append(data)

                plt.axis([0, 192, floor, ceil])
                plt.plot(data, linewidth=0.5)
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)

                filename = output_dir + 'fig_{}'.format(count) + '.png'
                plt.savefig(filename)
                plt.close()
                im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.erode(im_gray, kernel, iterations=1)
                im_gray = cv2.resize(im_gray, (192, 128), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(filename, im_gray)
                print('img writtten {}'.format(filename))
                count += 1


    return results



if __name__ == "__main__":
    records = get_records()
    """'N' for normal beats. Similarly we can give the input 'L' for left bundle branch block beats. 'R' for right bundle branch block
        beats. 'A' for Atrial premature contraction. 'V' for premature ventricular contraction. '/' for paced beat. 'E' for Ventricular
        escape beat."""
    labels = ['N', 'L', 'R', 'A', 'V', '/', 'E', '!']
    output_dirs = ['NOR/', 'LBBB/', 'RBBB/', 'APC/', 'PVC/', 'PAB/', 'VEB', 'VFE']

    for type, output_dir in zip(labels, output_dirs):
        sgs = segmentation(records, type, output_dir='./MIT-BIH_AD/'+output_dir)

