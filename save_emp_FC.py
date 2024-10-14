import utils
import os
import pandas as pd
import numpy as np

def save_emp_FC(folder, suffix, directory = os.path.dirname(__file__)):
    FC_ar = []
    TS_path = utils.get_folder(folder, directory)
    TS_matrix_ar = utils.matrix_from_dir(TS_path)

    for ar in TS_matrix_ar:
        FC_ar.append(np.corrcoef(ar.T))

    avg_TS = utils.average_matrices(*FC_ar)
    avg_TS = pd.DataFrame(avg_TS)
    avg_TS.to_csv('avg_TS_{}'.format(suffix), index = False, header = False)


if __name__ == '__main__':
    save_emp_FC('TS_1', '1', utils.get_folder('TS_data'))
    save_emp_FC('TS_2', '2', utils.get_folder('TS_data'))
    save_emp_FC('TS_3', '3', utils.get_folder('TS_data'))
