"""
author: li you wang
date: 11/12/2020

"""
import numpy as np
import cv2


def calculate_scale(template_points, register_points):
    """
    计算两个点集之间的scale
    :param template_points:
    :param register_points:
    :return:
    """
    scale = 1
    return scale


def icp_svd(template_points, register_points):
    """
    通过svd分解计算两个点集之间的旋转矩阵和平移向量
    :param template_points:
    :param register_points:
    :return:
    """


def icp_quaternion(template_points, register_points):
    """
    计算两个点集之间的icp 配准
    参考https://blog.csdn.net/hongbin_xu/article/details/80537100
    博客中的旋转矩阵写错了,需要按照论文里面的修改 而且要转转置
    :param template_points: 模板点集 N 3
    :param register_points: 带配准点集 N 3
    :return:
    """
    row, col = template_points.shape
    mean_template_points = np.mean(template_points, axis=0)
    mean_register_points = np.mean(register_points, axis=0)
    cov = (template_points - mean_template_points).T.dot((register_points - mean_register_points)) / row
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
    rotation_matrix = np.array([[np.square(q[0]) + np.square(q[1]) - np.square(q[2]) - np.square(q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
                                [2 * (q[1] * q[2] + q[0] * q[3]), np.square(q[0]) - np.square(q[1]) + np.square(q[2]) - np.square(q[3]), 2 * (q[2] * q[3] - q[0] * q[1])],
                                [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), np.square(q[0]) - np.square(q[1]) - np.square(q[2]) + np.square(q[3])]], dtype=np.float32).T
    print("旋转矩阵是{}".format(rotation_matrix))
    tranform_matrix = mean_template_points - rotation_matrix.dot(mean_register_points)
    print("计算出来的平移向量是{}".format(tranform_matrix))
    registered_points = rotation_matrix.dot(register_points.T).T + tranform_matrix
    error = np.mean(np.sqrt(np.sum(np.square(registered_points - template_points), axis=1)))
    print("align error is {}".format(error))


if __name__ == "__main__":
    test_points = np.random.rand(10, 3)
    rotation_vector = np.array([1, 1, 2], dtype=np.float32)
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    print("rotation_matrix is {}".format(rotation_matrix))
    transform_vector = np.array([2, 2, 2], dtype=np.float32)
    print("正确的平移向量是{}".format(transform_vector))
    register_points = rotation_matrix.dot(test_points.T).T + transform_vector
    icp_quaternion(register_points, test_points)
