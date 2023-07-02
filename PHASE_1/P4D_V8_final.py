import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from os import path

enc = LabelEncoder()

# ------------Reading columns from csv--------------
np.set_printoptions(formatter={'float_kind': lambda x: "%.9f" % x})
np.seterr(divide='ignore', invalid='ignore')


# ------------plotting graphs---------
def plot_graph(matrix, y, x, mysimplane):
    plt.figure()
    first_col = matrix[:, [0]]
    first_col = first_col.ravel()

    second_col = matrix[:, [1]]
    second_col = second_col.ravel()

    third_col = matrix[:, [2]]
    third_col = third_col.ravel()

    fourth_col = matrix[:, [3]]
    fourth_col = fourth_col.ravel()

    fig = plt.figure()

    ax = fig.add_subplot(221, projection='3d')
    ax.plot_trisurf(y, x, first_col, color='white', edgecolors='grey', alpha=0.1)
    ax.scatter(y, x, first_col, c='orange')
    ax.plot_trisurf(y, x, mysimplane, color='black', edgecolors='grey', alpha=0.1)
    ax.scatter(y, x, mysimplane, c='blue')
   
    ax = fig.add_subplot(222, projection='3d')
    ax.plot_trisurf(y, x, second_col, color='white', edgecolors='grey', alpha=0.1)
    ax.scatter(y, x, second_col, c='pink')
    ax.plot_trisurf(y, x, mysimplane, color='black', edgecolors='grey', alpha=0.1)
    ax.scatter(y, x, mysimplane, c='blue')

    ax = fig.add_subplot(223, projection='3d')
    ax.plot_trisurf(y, x, third_col, color='white', edgecolors='grey', alpha=0.1)
    ax.scatter(y, x, third_col, c='yellow')
    ax.plot_trisurf(y, x, mysimplane, color='black', edgecolors='grey', alpha=0.1)
    ax.scatter(y, x, mysimplane, c='blue')

    ax = fig.add_subplot(224, projection='3d')
    ax.plot_trisurf(y, x, fourth_col, color='white', edgecolors='grey', alpha=0.1)
    ax.scatter(y, x, fourth_col, c='green')
    ax.plot_trisurf(y, x, mysimplane, color='black', edgecolors='grey', alpha=0.1)
    ax.scatter(y, x, mysimplane, c='blue')

    plt.show()
    # ------------Used to save graphs in partivular folder ----------------
    #outpath = "B:/MTech Dissertation/Implementation/HYBRID MODEL/PHASE_1/ ------------/"
    #fig.savefig(path.join(outpath,"graph__onlineretail.png"))
   
    
def get_rotation_matrix(x, y):
    rotation_matrix = [[math.cos(math.radians(y)), (-1) * math.sin(math.radians(y)), 0, 0],
                       [math.sin(math.radians(y)), math.cos(math.radians(y)), 0, 0],
                       [0, 0, math.cos(math.radians(x)), (-1) * math.sin(math.radians(x))],
                       [0, 0, math.sin(math.radians(x)), math.cos(math.radians(x))]]

    return rotation_matrix


def create_set(cols, df):
    return np.column_stack(list(map(lambda column: list(df[column]),
                                    cols)))


def save_set(arr_set, rotation_matrix, new_file, rem_cols=0):
    arr_set = arr_set.transpose()
    foo = np.matmul(rotation_matrix, arr_set)
    if rem_cols:
        foo = foo[-rem_cols:, :]
    np.savetxt(new_file, foo, delimiter=',')


def main():
    # -------------Program starts from this point-------------------
    filename = input('please enter input csv filename: ')
    df = pd.read_csv(filename, delimiter=',', encoding='utf-8')

    all_cols = df.columns

    for col in all_cols:
        if is_string_dtype(df[col]):
            # Convert non numeric to numeric first
            df[col] = df[col].apply(lambda x: int.from_bytes(x.encode(), 'little'))

        # Normalize the data
        min_val = df[col].min()
        max_val = df[col].max()
        diff = max_val - min_val
        df[col] = df[col].apply(lambda x: ((x - min_val) / diff) * 5)

    col_len = len(all_cols)
    end_index = 4
    all_sets = []
    set_no = 1
    while end_index <= col_len:
        cols = all_cols[end_index - 4:end_index]
        arr_set = create_set(cols, df)
        all_sets.append(arr_set)
        print("Set No. {}: {}".format(set_no,
                                      ','.join(cols)))
        end_index += 4
        set_no += 1
    remaining_cols = col_len % 4
    if remaining_cols:
        cols = all_cols[col_len - 4:col_len]
        arr_set = create_set(cols, df)
        all_sets.append(arr_set)
        print("Set No. {}: {}".format(set_no,
                                      ','.join(cols)))

    sec_values = input('Enter comma separated threshold values for {} sets: '
                       .format(len(all_sets))).split(',')
    sec_values = list(map(float, sec_values))
    max_var, final_x, final_y = 0, 0, 0
    gen_y = list(range(0, 360, 4))
    gen_x = list(range(0, 360, 4))

    for sec_val, arr_set in zip(sec_values, all_sets):
        arr_set = arr_set.transpose()
        myx = []
        myy = []
        matrix = []
        my_simple_plane = []

        for y in gen_x:
            for x in gen_y:
                myx.append(x)
                myy.append(y)
                my_simple_plane.append(sec_val)
                rotation_matrix = get_rotation_matrix(x, y)
                foo = np.matmul(rotation_matrix, arr_set)
                sub_val = np.subtract(arr_set, foo)  # ---------subtract from original
                sub_val = sub_val.transpose()
                var_val = np.var(sub_val, axis=0)  # ----------variance along y axis
                if x == 0 and y == 0:
                    matrix = [var_val]
                else:
                    matrix = np.concatenate((matrix, [var_val]), axis=0)

                if max(var_val) > max_var:
                    final_x = x
                    final_y = y
                    max_var = max(var_val)

        plot_graph(matrix, myy, myx, my_simple_plane)

    print("Final values of x: {}".format(final_x))
    print("Final values of y: {}".format(final_y))

    rotation_matrix = get_rotation_matrix(final_x, final_y)

    filename = 'arr_set_new.csv'
    new_file = open(filename, 'w')
    new_file.close()
    new_file = open(filename, 'ab')

    for arr_set in all_sets[:-1]:
        save_set(arr_set, rotation_matrix, new_file)

    save_set(all_sets[-1], rotation_matrix, new_file, remaining_cols)
    new_file.close()

    f = open(filename, 'rb')
    foo = np.loadtxt(f, delimiter=',')
    f = open(filename, 'wb')
    np.savetxt(f, foo.transpose(), delimiter=',')


if __name__ == '__main__':
    main()
