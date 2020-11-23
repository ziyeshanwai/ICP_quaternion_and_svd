import cv2
import numpy as np
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


def transform_with_scale_quaternion(template_points, register_points):
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
    t = t[:, np.newaxis]
    error = np.mean(np.sum(np.square(template_points - (s * r.dot(register_points.T) + t).T), axis=1))
    return s, r, t, error


def transform_with_scale_svd(template_points, register_points):
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
    return s, r, t, error


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
        s, R, t, error = transform_with_scale_svd(tem, data)
        for i in range(max_iteration):
            transformed_points = (s * R.dot(register_points.T) + t).T
            correspondence, ind = update_correspondence(transformed_points, template_points)
            s, R, t, error = transform_with_scale_svd(template_points, register_points[ind, :])
            if error < 0.1:
                print("iteration {} stopped".format(i))
                break
        print("iteration over")
        return s, R, t, error

    if method == 'quaternion':
        tem, data = ini_correspondence
        s, R, t, error = transform_with_scale_quaternion(tem, data)
        for i in range(max_iteration):
            transformed_points = (s * R.dot(register_points.T) + t).T
            correspondence, ind = update_correspondence(transformed_points, template_points)
            s, R, t, error = transform_with_scale_quaternion(template_points, register_points[ind, :])
            if error < 0.1:
                print("iteration {} stopped".format(i))
                break
        print("iteration over")
        return s, R, t, error
        pass


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


def test_0():
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


def test_1():
    template_points, register_points, ini_correspondence = initial_matching()
    mu, sigma = 0, 0.1
    white_noise = np.random.normal(mu, sigma, size=(register_points.shape[0], 3))
    register_points = register_points + white_noise  # add noise
    icp(template_points, register_points, ini_correspondence, method='svd', max_iteration=10)
    icp(template_points, register_points, ini_correspondence, method='quaternion', max_iteration=10)


if __name__ == '__main__':
    test_1()