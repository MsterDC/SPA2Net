import matplotlib.pyplot as plt 
import os
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



def plot_headmap(data):
    sns.set()
    # p1 = sns.heatmap(data, square=True, annot=True)
    p1 = sns.heatmap(data)



if __name__=="__main__":

    root_path = 'C:\\Users\\l\\Desktop'
    _prefix = 'Black_Footed_Albatross_0046_18'
    _middle_sa = '_sa_h'
    _middle_hpsa = '_hpsa_h'
    _suffix = '.npy'

    hsc_file_id = _prefix + _suffix
    sa_id = _prefix + _middle_sa
    hpsa_id = _prefix + _middle_hpsa

    head_num = 8
    row_num = 4
    col_num = 5

    sa_plot_pos = [2,3,4,5,7,8,9,10]
    hpsa_plot_pos = [12,13,14,15,17,18,19,20]

    plt.figure(figsize=(18,12))

    hsc_file_path = os.path.join(root_path, hsc_file_id)
    hsc_np_file = np.load(hsc_file_path)

    hsc_pos_id = str(row_num) + str(col_num) + str(1)
    hsc_pos_id = int(hsc_pos_id)
    plt.subplot(hsc_pos_id)
    plt.axis('off')
    plt.title('HSC')
    plt.tight_layout()
    plot_headmap(hsc_np_file)
    
    for i in range(head_num):

        # sa and hpsa map
        sa_file_id = sa_id + str(i) + _suffix
        hpsa_file_id = hpsa_id + str(i) + _suffix

        sa_file_path = os.path.join(root_path, sa_file_id)
        hpsa_file_path = os.path.join(root_path, hpsa_file_id)

        sa_np_file = np.load(sa_file_path)
        hpsa_np_file = np.load(hpsa_file_path)

        plt.subplot(row_num,col_num,sa_plot_pos[i])
        plt.axis('off')
        plt.title('h=' + str(i))
        plt.tight_layout()
        plot_headmap(sa_np_file)

        plt.subplot(row_num, col_num, hpsa_plot_pos[i])
        plt.axis('off')
        plt.title('h=' + str(i))
        plt.tight_layout()
        plot_headmap(hpsa_np_file)


    plt.show()

    print("Success!")