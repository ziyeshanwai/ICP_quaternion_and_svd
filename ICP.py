"""
author: li you wang
date: 11/12/2020

"""
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in src for each point in dst
    Input:
        src: template points
        dst: destination points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor with dst
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(src)
    distances, indices = neigh.kneighbors(dst, return_distance=True)
    return distances.ravel(), indices.ravel()


def transform_svd(template_points, register_points):
    """
    使用svd分解得到旋转矩阵
    :param template_points: N 3
    :param register_points: N 3
    :return:
    """
    print('-' * 100)
    print("icp based on svd")
    row, col = template_points.shape
    w = np.eye(row)
    p = register_points.T
    q = template_points.T
    mean_p = p.dot(np.diagonal(w).reshape(-1, 1)) / np.trace(w)
    mean_q = q.dot(np.diagonal(w).reshape(-1, 1)) / np.trace(w)
    X = p - mean_p
    Y = q - mean_q
    S = X.dot(w).dot(Y.T)
    U, sigma, VT = np.linalg.svd(S)
    det_v_ut = np.linalg.det(VT.T.dot(U.T))
    diag_matrix = np.eye(3)
    diag_matrix[2, 2] = det_v_ut
    rotation_matrix = VT.T.dot(diag_matrix).dot(U.T)
    print("旋转矩阵是{}".format(rotation_matrix))
    transformation_matrix = mean_q - rotation_matrix.dot(mean_p)
    print("平移矩阵是{}".format(transformation_matrix))
    error = np.mean(np.sqrt(np.sum(np.square((rotation_matrix.dot(register_points.T) + transformation_matrix).T - template_points), axis=1)))
    print("error is {}".format(error))
    R = rotation_matrix
    t = transformation_matrix
    return R, t, error


def quaternion_2_rotation_matrix(q):
    """
    四元数转化为旋转矩阵
    :param q:
    :return: 旋转矩阵
    """
    rotation_matrix = np.array([[np.square(q[0]) + np.square(q[1]) - np.square(q[2]) - np.square(q[3]),
                                 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
                                [2 * (q[1] * q[2] + q[0] * q[3]),
                                 np.square(q[0]) - np.square(q[1]) + np.square(q[2]) - np.square(q[3]),
                                 2 * (q[2] * q[3] - q[0] * q[1])],
                                [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]),
                                 np.square(q[0]) - np.square(q[1]) - np.square(q[2]) + np.square(q[3])]],
                               dtype=np.float32)
    return rotation_matrix


def rotation_matrix_2_quaternion(rotation_matrix):
    """
    旋转矩阵转化为四元数
    :param rotation_matrix:
    :return:
    """
    R = rotation_matrix
    q0 = np.sqrt(np.trace(R) + 1) / 2
    q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
    q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
    q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    quaternion = np.array([q0, q1, q2, q3], dtype=np.float32)
    print("四元数的模是{}".format(np.linalg.norm(quaternion)))
    return quaternion


def transform_quaternion(template_points, register_points):
    """
    计算两个点集之间的icp 配准
    参考https://blog.csdn.net/hongbin_xu/article/details/80537100
    博客中的旋转矩阵写错了,需要按照论文里面的修改
    :param template_points: 模板点集 N 3
    :param register_points: 带配准点集 N 3
    :return:
    """
    print('-'*100)
    print("icp based on quaternion")
    row, col = template_points.shape
    mean_template_points = np.mean(template_points, axis=0)
    mean_register_points = np.mean(register_points, axis=0)
    cov = (register_points - mean_register_points).T.dot((template_points - mean_template_points)) / row
    A = cov - cov.T
    delta = np.array([A[1, 2], A[2, 0], A[0, 1]], dtype=np.float32).T
    Q = np.zeros((4, 4), dtype=np.float32)
    Q[0, 0] = np.trace(cov)
    Q[0, 1:] = delta
    Q[1:, 0] = delta
    Q[1:, 1:] = cov + cov.T - np.trace(cov)*np.eye(3)
    lambdas, vs = np.linalg.eig(Q)
    q = vs[:, np.argmax(lambdas)]
    print("四元数是{}".format(q))
    rotation_matrix = quaternion_2_rotation_matrix(q)
    q_hat = rotation_matrix_2_quaternion(rotation_matrix)
    print("反算的四元数是{}".format(q_hat))
    print("旋转矩阵是{}".format(rotation_matrix))
    tranform_matrix = mean_template_points - rotation_matrix.dot(mean_register_points)
    print("计算出来的平移向量是{}".format(tranform_matrix))
    registered_points = rotation_matrix.dot(register_points.T).T + tranform_matrix
    error = np.mean(np.sqrt(np.sum(np.square(registered_points - template_points), axis=1)))
    print("align error is {}".format(error))
    R = rotation_matrix
    t = tranform_matrix[:, np.newaxis]
    return R, t, error


def print_statics(points):
    """
    print statics
    :param points:
    :return:
    """
    test_points_mean = np.mean(points, axis=0)
    test_points_var = np.var(points, axis=0)
    test_points_std = np.std(points, axis=0)
    print("points mean var std is {}, {}, {}".format(test_points_mean, test_points_var, test_points_std))


def update_correspondence(src, dst):
    """
    build correspondence between src and dst
    :param src: template points N 3
    :param dst: register points M 3
    :return: correspondence
    """
    dis, ind = nearest_neighbor(src, dst)  # the length of ind is M
    correspondence = (src[ind, :], dst)
    return correspondence, ind


def initial_matching():
    """
    指定初始匹配关系
    :param template_points:
    :param register_points:
    :return:
    """
    n_number = 20
    register_points = np.random.rand(n_number, 3)
    rotation_vector = np.array([1, 1, 2], dtype=np.float32)
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    print("gt rotation_matrix is {}".format(rotation_matrix))
    transform_vector = np.array([2, 2, 2], dtype=np.float32)
    tranformed_points = rotation_matrix.dot(register_points.T).T + transform_vector
    print("gt translation_vector is {}".format(transform_vector))
    choosed_index = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]) # 选择的点
    template_points = tranformed_points[choosed_index, :]
    correspondence = (template_points, register_points[choosed_index, :])
    return template_points, register_points, correspondence


def icp(template_points, register_points, ini_correspondence, method, max_iteration=100):
    """
    icp with different method 'svd' 'quaternion'
    :param template_points:
    :param register_points:
    :param max_iteration:
    :param method:
    param ini_correspondence: 初始对应关系 利用初始对应关系即可算的一个初始的r t
    :return: best rotation matrix and translation vector
    """
    if method == 'svd':
        tem, data = ini_correspondence
        R, t, error = transform_svd(tem, data)
        for i in range(max_iteration):
            transformed_points = (R.dot(register_points.T) + t).T
            correspondence, ind = update_correspondence(transformed_points, template_points)
            R, t, error = transform_svd(template_points, register_points[ind, :])
            if error < 0.1:
                print("iteration {} stopped".format(i))
                break
        print("iteration over")
        return R, t, error

    if method == 'quaternion':
        tem, data = ini_correspondence
        R, t, error = transform_quaternion(tem, data)
        for i in range(max_iteration):
            transformed_points = (R.dot(register_points.T) + t).T
            correspondence, ind = update_correspondence(transformed_points, template_points)
            R, t, error = transform_quaternion(template_points, register_points[ind, :])
            if error < 0.1:
                print("iteration {} stopped".format(i))
                break
        print("iteration over")
        return R, t, error
        pass


def test():
    template_points, register_points, ini_correspondence = initial_matching()
    mu, sigma = 0, 0.1
    white_noise = np.random.normal(mu, sigma, size=(register_points.shape[0], 3))
    register_points = register_points + white_noise  # add noise
    # icp(template_points, register_points, ini_correspondence, method='svd', max_iteration=10)
    icp(template_points, register_points, ini_correspondence, method='quaternion', max_iteration=10)


def test_1():
    n_number = 4
    test_points = np.random.rand(n_number, 3)
    print_statics(test_points)
    rotation_vector = np.array([1, 1, 2], dtype=np.float32)
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    print("gt rotation_matrix is {}".format(rotation_matrix))
    print("gt quaternion is {}".format(rotation_matrix_2_quaternion(rotation_matrix)))
    transform_vector = np.array([2, 2, 2], dtype=np.float32)
    print("gt translation vector {}".format(transform_vector))
    register_points = rotation_matrix.dot(test_points.T).T + transform_vector
    print_statics(register_points)
    mu, sigma = 0, 0.1
    white_noise = np.random.normal(mu, sigma, size=(n_number, 3))
    register_points = register_points + white_noise  # add noise
    r, t, error = transform_quaternion(register_points, test_points)
    rotation_vector = cv2.Rodrigues(r)[0]
    print("rotation_vector is {}".format(rotation_vector))
    transform_svd(register_points, test_points)


if __name__ == "__main__":
    test()
    test_1()

