import utils
import numpy as np


def avg_Jij(folder):
    folder_path = utils.get_folder(folder)
    matrix_ar = utils.matrix_from_dir(folder_path)
    return utils.average_matrices(*matrix_ar)


def avg_Jij_no_outliers(folder):
    folder_path = utils.get_folder(folder)
    matrix_ar = utils.matrix_from_dir(folder_path)
    std_ar = np.std(matrix_ar, axis = 0)
    length = range(np.shape(std_ar)[0])

    Jij = np.zeros(np.shape(std_ar))
    for y in length:
        for x in length:
            matrix = matrix_ar[:, y, x]
            #print(matrix[matrix >= std_ar[y, x]], std_ar[y, x])
            Jij[y, x] = np.mean(matrix[matrix >= std_ar[y, x]])

    return Jij


if __name__ == '__main__':
    utils.save_matrix(avg_Jij_no_outliers('Jij_data'), 'avg_Jij_no_outliers')
    utils.save_matrix(avg_Jij('Jij_data'), 'avg_Jij')
