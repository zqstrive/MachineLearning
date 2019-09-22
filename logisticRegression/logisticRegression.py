import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
# from tqdm import tqdm
# import time


# transform data file to csv file
def data_to_csv():
    import pandas as pd
    path1 = './data/iris.data'
    data = pd.read_table(path1, header=None, sep=',')
    print(data)
    data = data.to_csv('./data/iris.csv', index=False, header=False)
    print(data)


# load data
def load_data(file_name):
    """
    :param file_name: name of dataset
    :return: array/matrix corresponding to dataset
    """
    # four features of dataset
    # x_features = {"sepal length", "sepal width", "petal length", "petal width"}
    # the location of dataset
    file_name = file_name
    # set the precision is 1,to keep same to dataset
    np.set_printoptions(precision=1)
    # take the first 100 rows of dataset
    # x_mat contains four features of X  x_mat->features
    x_mat = np.loadtxt(file_name, delimiter=',', usecols=[0, 1, 2, 3], dtype=np.float16)[:100]
    # y_label contains the label of X   yMat->label
    y_label = np.loadtxt(file_name, delimiter=',', usecols=[4], dtype=np.str)[:100]

    # convert str to int  1/0
    y_label[0:50] = 1
    y_label[50:] = 0
    y_label = y_label.astype(np.int32)

    # total dataset array   column merge
    data_mat = np.column_stack((x_mat, y_label))
    return data_mat


# sigmoid function
def sigmoid(z):
    """
    :param z: input
    :return: data after compression
    """
    y = 1.0 / (1 + np.exp(-z))
    return y


# PCA dimensionality reduction
def pca(x_mat, k):
    """
    :param x_mat:data to reduce dimensionality
    :param k: dimension that want to dimensionality reduction
    :return: dimensioned dataset,original dataset
    """
    x_mat = x_mat
    row, col = x_mat.shape
    # axis=0 compress row，compute average of column
    x_avg = np.mean(x_mat, axis=0)
    # row copy row times,column does not copy
    x_avg_mat = np.tile(x_avg, (row, 1))
    # centralized data
    x_adjust = x_mat - x_avg_mat
    # calculate the covariance matrix
    covX = np.cov(x_adjust.T)
    # calculate eigenvalues and eigenvectors
    feature_value, feature_vector = np.linalg.eig(covX)
    # Calculate the corresponding subscript according to the eigenvalue from large to small
    max_index = np.argsort(-feature_value)
    pca_mat = []  # dimensioned dataset
    start_mat = []  # original dataset
    if k > col:
        print('error,k must less than col')
    else:
        # The eigenvectors is column vectors, so it need to be transposed and intercepted
        select_vector = feature_vector.T[max_index[:k]]
        # calculate the projection matrix
        pca_mat = np.dot(x_adjust, select_vector.T)
        start_mat = (np.dot(pca_mat, select_vector)) + x_avg
    return pca_mat, start_mat


# calculate test set accuracy
def compute_accuracy(test_x, test_label, weights):
    """
    :param test_x: features of test set
    :param test_label: label of test set
    :param weights: weights
    :return: accuracy
    """
    test_x = test_x
    test_label = test_label
    test_num, test_features = test_x.shape
    # w,b together for easy calculation
    test0_mat = np.ones(test_num)
    # merge in the column direction
    test_x = np.column_stack((test_x, test0_mat))
    weights = weights
    # the num of correct predict
    count = 0
    # logisticRegression function，predict ouput
    test_predict = sigmoid(np.dot(test_x, weights.T))
    # print(test_predict)
    # ouput>0.5 is positive 1，on the contrary is negetive 0
    test_predict_label = [1 if elem > 0.5 else 0 for elem in test_predict]
    # print(test_predict_label)
    # print(test_label)
    # calculate accuracy
    for i in range(len(test_label)):
        if test_label[i] == test_predict_label[i]:
            count += 1
    print('test_accuracy = {0}%'.format(100 * (count / test_num)))
    return test_predict_label


# print as a function of the number of training increases, the curve of the model loss function
def cost_plot(costs, num_iter):
    """
    :param costs:costFunction get the updated value after 100 rounds
    :param num_iter: num of iterations
    :return: cost change graph with the number of iterations
    """
    costs = costs
    num_iter = num_iter
    plt.plot(np.arange(0, num_iter, 100), costs)
    plt.title("Cost Function after different iterations")
    plt.xlabel("number of iterations")
    plt.ylabel("Cost")
    plt.show()


# print the data after pca dimension reduction and the fitted line 2d
def data_view2d(pca_data_mat,weights):
    """
    :param pca_data_mat:dimensioned dataset
    :return:2d data distribution map
    """
    # row merge  np.concatenate(X,Y,axis=0)
    pca_data_mat = pca_data_mat
    row,col = pca_data_mat.shape
    positive_num = row//2
    extra_arr = np.ones(row)
    pca_data_mat_compute = np.column_stack((pca_data_mat,extra_arr))
    weights = weights
    # print(weights)
    # define array have two points
    arrs = []
    # row, col = pca_data_mat.shape
    feature_one_min = pca_data_mat[:, 0].min()
    feature_two_min = pca_data_mat[:, 1].min()
    feature_one_max = pca_data_mat[:, 0].max()
    feature_two_max = pca_data_mat[:, 1].max()

    # compute two points (x1,x2) (x1',x2') and draw a straight line
    feature_min_label = (-weights[2]-weights[0]*feature_one_min)/weights[1]
    feature_max_label = (-weights[2]-weights[0]*feature_one_max)/weights[1]
    arrs.append([feature_one_min,feature_min_label])
    arrs.append([feature_one_max,feature_max_label])
    plt.axis([feature_one_min, feature_one_max, feature_two_min, feature_two_max])
    plt.scatter(pca_data_mat[:positive_num, 0], pca_data_mat[:positive_num, 1], c='r', label='setosa')
    plt.scatter(pca_data_mat[positive_num:, 0], pca_data_mat[positive_num:, 1], c='g', label='versicolor')
    # print(arrs)
    plt.plot(arrs[0],arrs[1])
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title("Iris")
    plt.legend()
    plt.show()


# print 3d map
def data_view3d(pca_data_mat, pca_label_mat, test_predict_label):
    """
    :param pca_data_mat: dimensioned dataset
    :param pca_label_mat: dimensioned label set
    :param test_predict_label: predicted label after parameters updating
    :return: 3D scatter plot
    """
    pca_data_mat = pca_data_mat
    pca_label_mat = pca_label_mat
    # two features correspond to two axes
    feature_one = pca_data_mat[:, 0]
    feature_two = pca_data_mat[:, 1]
    y_label = pca_label_mat
    test_predict_label = test_predict_label
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(feature_one, feature_two, y_label, c=y_label)
    ax.scatter(feature_one, feature_two, test_predict_label, c='r')
    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.set_zlabel('class')
    plt.show()


# sgd update w
def stochastic_gradient_descent(x_mat, y_label, num_iter, learning_rate):
    """
    :param x_mat:dataset
    :param y_label: label set
    :param num_iter: number of iterations
    :param learning_rate: learning rate
    :return: w and cost
    """
    # cross
    cost = 0
    costs = []
    x_mat = x_mat
    x_nums, x_features = x_mat.shape
    # merge parameter for easy compute
    x0_mat = np.ones(x_nums)
    x_mat = np.column_stack((x_mat, x0_mat))
    y_label = y_label
    num_iter = num_iter
    learning_rate = learning_rate
    weights = np.random.randn(x_features + 1)
    # show loop of progress bar
    # for i in tqdm(range(num_iter)):
    for i in range(num_iter):
        # random select a sample to update weight
        rand_index = int(np.random.uniform(0, x_nums))
        y_predict = sigmoid(np.dot(x_mat[rand_index], weights.T))

        for j in range(x_features+1):
            error = -(y_label[rand_index] - y_predict)
            # for the concrete w_j update parameter
            weights[j] = weights[j] - learning_rate * error * x_mat[rand_index][j]

        # computer cost
        for k in range(x_nums):
            y_pre = sigmoid(np.dot(x_mat[k], weights.T))
            cost+=y_label[k]*np.log(y_pre)+(1-y_label[k])*np.log(1-y_pre)
        cost = -cost/x_nums
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
        # time.sleep(0.001)
    return weights, costs

# Adam update w
def Adam(x_mat, y_label, num_iter, learning_rate):
    """
    :param x_mat:dataset
    :param y_label: label set
    :param num_iter: number of iterations
    :param learning_rate: learning rate
    :return: w and cost
    """
    cost = 0
    costs = []
    x_mat = x_mat
    x_nums, x_features = x_mat.shape
    # merge parameter for easy compute
    x0_mat = np.ones(x_nums)
    x_mat = np.column_stack((x_mat, x0_mat))
    y_label = y_label
    num_iter = num_iter
    learning_rate = learning_rate
    weights = np.random.randn(x_features + 1)
    p1 = 0.9
    p2 = 0.999
    delta = 1e-8
    s = 0
    r = 0
    t = 0
    for i in range(num_iter):
        rand_index = int(np.random.uniform(0, x_nums))
        y_predict = sigmoid(np.dot(x_mat[rand_index], weights.T))

        for j in range(x_features+1):
            error = -(y_label[rand_index] - y_predict)
            g = error*x_mat[rand_index][j]
            t += 1
            s = p1*s+(1-p1)*g
            r = p2*r+(1-p2)*g*g
            s_hat = s/(1-p1**t)
            r_hat = r/(1-p2**t)
            weightsj_delta = -learning_rate*s_hat/(np.sqrt(r_hat)+delta)
            weights[j] += weightsj_delta

            # weights[j] = weights[j] - learning_rate * error * x_mat[rand_index][j]

        for k in range(x_nums):
            y_pre = sigmoid(np.dot(x_mat[k], weights.T))
            cost += y_label[k] * np.log(y_pre) + (1 - y_label[k]) * np.log(1 - y_pre)
        cost = -cost / x_nums
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
            # time.sleep(0.001)
    return weights, costs


# split dataset
def data_split(data_mat, test_rate):
    """
    :param data_mat:dataset
    :param test_rate: proportion of test set
    :return: train set features,train set label,test set features,test set label
    """
    data_mat = data_mat
    test_rate = test_rate
    row, col = data_mat.shape
    # number of train set
    test_num = row * test_rate
    # keep the label balance，compute positive num
    positive_num = test_num // 2
    train_set = []
    test_set = []
    # non-repeating select subscript
    random_positive_choice = random.sample(range(50), int(positive_num))
    random_negative_choice = random.sample(range(50, 100), int(positive_num))
    # randomly select test set
    for i in random_positive_choice:
        test_set.append(data_mat[i].tolist())
    for j in random_negative_choice:
        test_set.append(data_mat[j].tolist())

    # randomly select train set
    for i in range(50):
        if i in random_positive_choice:
            continue
        train_set.append(data_mat[i].tolist())
    for j in range(50, 100):
        if j in random_negative_choice:
            continue
        train_set.append(data_mat[j].tolist())

    # convert list to array
    test_set1 = np.array(test_set)
    train_set1 = np.array(train_set)

    # split data
    train_x = train_set1[:, :row]
    train_label = train_set1[:, -1]
    test_x = test_set1[:, :row]
    test_label = test_set1[:, -1]
    return train_x, train_label, test_x, test_label


# logisticRegression model
def logistic_regression(file_name, test_rate, k, num_iter, learning_rate,optimization):
    """
    :param file_name: data file location
    :param test_rate: proportion of test set
    :param k: dimension that want to dimensionality reduction
    :param num_iter: number of iterations
    :param learning_rate: learning rate
    :param optimization:optimization algorithm
    :return: logistic_regression model
    """
    print(
        '****************use {0}% of dataset as train set，{1}% of dataset as test set***********************'.format(100 * (1 - test_rate), 100 * test_rate))
    file_name = file_name
    test_rate = test_rate
    k = k
    num_iter = num_iter
    learning_rate = learning_rate
    optimization = optimization
    data_mat = load_data(file_name)
    feature_mat = data_mat[:,:4]
    label_mat = data_mat[:,-1]
    pca_data_mat, init_data_mat = pca(feature_mat, 2)
    # print(pca_data_mat)
    pca_data_mat1 = np.column_stack((pca_data_mat,label_mat))
    train_mat, train_label, test_mat, test_label = data_split(pca_data_mat1, test_rate)
    if optimization.strip() == 'sgd':
        weights, costs = stochastic_gradient_descent(train_mat, train_label, num_iter, learning_rate)
    elif optimization.strip() == 'Adam':
        weights, costs = Adam(train_mat, train_label, num_iter, learning_rate)
    test_predict_label = compute_accuracy(test_mat, test_label, weights)
    # # print(weights)
    cost_plot(costs, num_iter)
    data_view2d(test_mat,weights)
    # data_view3d(test_mat, test_label, test_predict_label)
    print('*****************************end*************************************')

def main():
    file_name = './data/iris.csv'
    logistic_regression(file_name, 0.5, 2, 10000, 0.001,'sgd')
    logistic_regression(file_name, 0.5, 2, 10000, 0.001,'Adam')
    # logistic_regression(file_name, 0.3, 2, 500, 0.1,'sgd')
    # logistic_regression(file_name, 0.1, 2, 500, 0.1,'sgd)


if __name__ == '__main__':
    main()
