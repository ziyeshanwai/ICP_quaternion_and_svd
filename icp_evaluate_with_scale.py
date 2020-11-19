import cv2
import numpy as np


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


def icp_with_scale_quaternion(template_points, register_points):
    """
    icp with scale using quaternion
    :param template_points:
    :param register_points:
    :return:
    """
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
    Q[1:, 1:] = cov + cov.T - np.trace(cov) * np.eye(3)
    lambdas, vs = np.linalg.eig(Q)
    q = vs[:, np.argmax(lambdas)]
    print("四元数是{}".format(q))
    rotation_matrix = quaternion_2_rotation_matrix(q)
    register_points_rotation = rotation_matrix.dot((register_points - mean_register_points).T)
    s = np.trace((template_points - mean_template_points).dot(register_points_rotation))/np.trace(register_points_rotation.T.dot(register_points_rotation))
    r = rotation_matrix
    t = mean_template_points - s * r.dot(mean_register_points)
    return s, r, t


def icp_with_scale(template_points, register_points):
    """
    icp with scale please pay attention to the potion of s r t
    计算data匹配到模板的s r t
    核心思想 尺度不会影响旋转矩阵 所以可以先求取旋转矩阵 然后计算出s
    reference: Point Set Registration with Integrated Scale Estimation
    :param template: 模板点 n 3
    :param data: 待配准点 n 3
    :return: s r t
    """
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
    r = rotation_matrix
    register_points_hat = rotation_matrix.dot(X).T
    print("rotation matrix is {}".format(rotation_matrix))
    s = np.trace(Y.T.dot(register_points_hat.T))/np.trace(register_points_hat.dot(register_points_hat.T))
    print("s is {}".format(s))
    t = mean_q - s * r.dot(mean_p)
    error = np.mean(np.sum(np.square(template_points - (s * r.dot(register_points.T) + t).T), axis=1))
    print("align error is {}".format(error))
    return s, r, t


if __name__ == '__main__':
    n_number = 4
    scale = 5
    register_points = np.random.rand(n_number, 3)
    rotation_vector = np.array([1, 1, 2], dtype=np.float32)
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    print("gt rotation_matrix is {}".format(rotation_matrix))
    transform_vector = np.array([2, 2, 2], dtype=np.float32)
    print("gt tranformation vector {}".format(transform_vector))
    template_points = rotation_matrix.dot(register_points.T).T * scale + transform_vector
    print("gt scale is {}".format(scale))
    # s, r, t = icp_with_scale(template_points, register_points)
    s, r, t = icp_with_scale_quaternion(template_points, register_points)
    print("prediction s r t is {}\n {}\n {}\n".format(s, r, t))