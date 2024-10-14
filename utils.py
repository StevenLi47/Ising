import numpy as np
import pandas as pd
import os


def get_folder(folder_name, directory = os.path.dirname(__file__)):
    cur_dir = directory
    return os.path.join(cur_dir, folder_name)


def get_matrix(filename, directory = os.path.dirname(__file__)):
    path = os.path.join(directory, filename)
    with open(path, newline='') as csvfile:
        return np.genfromtxt(csvfile, delimiter = ',')


def save_matrix(matrix, name):
    dataframe = pd.DataFrame(matrix)
    dataframe.to_csv(name, index = False, header = False)


def matrix_from_dir(directory):
    files = os.listdir(directory)
    matrix_ar = []

    for filename in files:
        file_path = get_matrix(os.path.join(directory, filename))
        matrix_ar.append(file_path)

    return np.array(matrix_ar)


def minmax_norm(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def normalize_array(array):
    return array / np.max(array)


def average_matrices(*arrays):
    avg_ar = np.zeros(np.shape(arrays[0]))

    for ar in arrays:
        avg_ar += ar

    avg_ar /= len(arrays)
    return avg_ar


def flat_remove_diag(array):
    new_ar = []
    length = range(np.shape(array)[0])
    for y in length:
        for x in length:
            if x != y:
                new_ar.append(array[y, x])
    return np.array(new_ar)

