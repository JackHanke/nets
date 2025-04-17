import numpy as np

# linear regression by direct computation using pseudoinverses of data matrix
def linreg(data: list):
    # initialize data matrix
    X = []
    # initialize labels matrix
    b = []
    n = len(data)
    for i in range(n):
        data_x = data[i][0]
        data_y = data[i][1]
        # make row of data matrix
        X.append([1, data_x])
        # make row of labels matrix
        b.append([data_y])

    X = np.array(X)
    b = np.array(b)
    # compute intermediate values for pseudoinverse
    X_t = np.linalg.matrix_transpose(X)
    prod_inv = np.linalg.inv(X_t @ X)

    # directly compute optimal coefficients
    coefficients = prod_inv @ X_t @ b

    return coefficients


