import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.linear_assignment_ import linear_assignment


def compute_iou(bbox, candidates):
    """
    计算交并比（IoU）。
    :param bbox: 边界框，格式为[x1, y1, x2, y2]
    :param candidates: 候选边界框列表
    :return: IoU 矩阵
    """
    x1 = np.maximum(bbox[0], candidates[:, 0])
    y1 = np.maximum(bbox[1], candidates[:, 1])
    x2 = np.minimum(bbox[2], candidates[:, 2])
    y2 = np.minimum(bbox[3], candidates[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + \
            (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1]) - intersection

    return intersection / union


def compute_cost_matrix(tracks, detections, feature_extractor):
    """
    计算代价矩阵。
    :param tracks: 轨迹列表
    :param detections: 检测列表
    :param feature_extractor: 特征提取器
    :return: 成本矩阵
    """
    cost_matrix = np.zeros((len(tracks), len(detections)))

    for t, track in enumerate(tracks):
        for d, detection in enumerate(detections):
            if track.is_confirmed() and not track.is_deleted():
                iou = compute_iou(track.to_tlwh(), detection.to_tlwh())
                cost_matrix[t, d] = iou

    for t, track in enumerate(tracks):
        for d, detection in enumerate(detections):
            if track.track_id == -1 or track.time_since_update > 1:
                feature = feature_extractor(track.features)
                cost_matrix[t, d] = feature[track.track_id] * feature[d]

    return cost_matrix


class Track:
    def __init__(self, mean, covariance, feature, track_id):
        self.mean = mean
        self.covariance = covariance
        self.feature = feature
        self.track_id = track_id
        self.time_since_update = 0

    def is_confirmed(self):
        return self.track_id != -1

    def is_deleted(self):
        return self.track_id == -2

    def to_tlwh(self):
        return self.mean[:4]

    def predict(self):
        self.mean = self.mean[0:2] + self.mean[2:4]
        self.covariance = np.diag(self.covariance.diagonal() * [1, 1, 0.1, 0.1])
        self.time_since_update += 1

    def update(self, new_mean, new_covariance, new_feature):
        self.mean = new_mean
        self.covariance = new_covariance
        self.feature = new_feature
        self.time_since_update = 0


class KalmanFilter:
    def __init__(self):
        self.transition_matrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        self.observation_matrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
        self.init_state = np.zeros(4)

    def predict(self, mean, covariance):
        mean = self.transition_matrix @ mean
        covariance = self.transition_matrix @ covariance @ self.transition_matrix.T
        return mean, covariance

    def update(self, mean, covariance, observation):
        predicted_mean, predicted_covariance = self.predict(mean, covariance)
        innovation = observation - predicted_mean
        innovation_covariance = self.observation_matrix @ predicted_covariance @ self.observation_matrix.T
        kalman_gain = predicted_covariance @ self.observation_matrix.T @ np.linalg.inv(innovation_covariance)
        new_mean = predicted_mean + kalman_gain @ innovation
        new_covariance = predicted_covariance - kalman_gain @ self.observation_matrix @ predicted_covariance
        return new_mean, new_covariance


class FeatureExtractor:
    def __init__(self):
        pass

    def extract(self, detection):
        return np.random.rand(128)


def linear_assignment(cost_matrix):
    """
    使用匈牙利算法进行线性和最小化。
    :param cost_matrix: 成本矩阵
    :return: 匹配索引
    """
    matched_indices = linear_assignment(-cost_matrix)
    return matched_indices


def update_tracks(tracks, detections, feature_extractor, kalman_filter):
    """
    更新轨迹。
    :param tracks: 轨迹列表
    :param detections: 检测列表
    :param feature_extractor: 特征提取器
    :param kalman_filter: 卡尔曼滤波器
    """
    cost_matrix = compute_cost_matrix(tracks, detections, feature_extractor)
    matched_indices = linear_assignment(cost_matrix)

    for i, j in matched_indices:
        if cost_matrix[i, j] > 0.5:
            track = tracks[i]
            detection = detections[j]
            observation_matrix = np.concatenate([detection.to_tlwh(), np.zeros((4, 1))], axis=1)
            mean, covariance = kalman_filter.update(track.mean, track.covariance, observation_matrix)
            track.update(mean, covariance, feature_extractor.extract(detection))

    for i, j in enumerate(tracks):
        if j.track_id == -1:
            j.update(mean, covariance, feature_extractor.extract(detections[i]))


# 示例数据
tracks = [Track(np.array([0, 0, 10, 10]), np.eye(4), feature_extractor.extract(detection), i) for i in range(5)]
detections = [detection for detection in range(5)]

# 特征提取器和卡尔曼滤波器
feature_extractor = FeatureExtractor()
kalman_filter = KalmanFilter()

# 更新轨迹
update_tracks(tracks, detections, feature_extractor, kalman_filter)