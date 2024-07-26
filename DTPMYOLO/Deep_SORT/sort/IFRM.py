import numpy as np
import cv2
from scipy.spatial.distance import cdist


def calculate_head_bbox(body_bbox, head_body_ratio=7.0):
    """
    计算头部边界框。
    :param body_bbox: 行人边界框，格式为[x, y, w, h]
    :param head_body_ratio: 头身比
    :return: 头部边界框
    """
    head_width = body_bbox[2] / (3 * head_body_ratio)
    head_height = body_bbox[3] / head_body_ratio
    head_center_x = body_bbox[0] + body_bbox[2] / 2
    head_center_y = body_bbox[1] + body_bbox[3] / 2 - head_height / 2

    head_bbox = [
        head_center_x - head_width / 2,
        head_center_y - head_height / 2,
        head_width,
        head_height
    ]
    return head_bbox


def match_masks_with_pedestrians(masks, nomask, pedestrians, head_body_ratio=7.0):
    """
    使用匈牙利算法将口罩和未佩戴口罩的边界框与行人边界框进行匹配。
    :param masks: 佩戴口罩的边界框列表
    :param nomask: 未佩戴口罩的边界框列表
    :param pedestrians: 行人边界框列表
    :param head_body_ratio: 头身比
    :return: 标记后的行人列表
    """
    matched_pedestrians = []

    # 计算行人头部边界框
    head_bboxes = [calculate_head_bbox(bbox, head_body_ratio) for bbox in pedestrians]

    # 计算代价矩阵
    cost_matrix = np.zeros((len(masks) + len(nomask), len(pedestrians)))

    for i, (mask_bbox, nomask_bbox) in enumerate(zip(masks, nomask)):
        for j, head_bbox in enumerate(head_bboxes):
            # 使用欧氏距离作为代价
            cost_matrix[i, j] = np.sqrt((mask_bbox[0] - head_bbox[0]) ** 2 + (mask_bbox[1] - head_bbox[1]) ** 2)

    for i, (mask_bbox, nomask_bbox) in enumerate(zip(masks, nomask)):
        for j, head_bbox in enumerate(head_bboxes):
            cost_matrix[len(masks) + i, j] = np.sqrt(
                (nomask_bbox[0] - head_bbox[0]) ** 2 + (nomask_bbox[1] - head_bbox[1]) ** 2)

    # 使用匈牙利算法进行匹配
    matched_indices = linear_sum_assignment(cost_matrix)

    for (mask_index, ped_index), (nomask_index, ped_index) in matched_indices:
        if mask_index < len(masks):
            matched_pedestrians.append((pedestrians[ped_index], 1))  # 佩戴口罩
        elif nomask_index < len(nomask):
            matched_pedestrians.append((pedestrians[ped_index], 0))  # 未佩戴口罩
        else:
            matched_pedestrians.append((pedestrians[ped_index], -1))  # 佩戴口罩情况未知

    return matched_pedestrians


def linear_sum_assignment(cost_matrix):
    """
    使用匈牙利算法计算线性和最小化。
    :param cost_matrix: 成本矩阵
    :return: 匹配索引
    """
    cost_matrix = cost_matrix.astype(np.float64)
    D = cost_matrix.min(axis=1)
    cost_matrix -= D[:, np.newaxis]
    D = cost_matrix.min(axis=0)
    cost_matrix -= D[np.newaxis, :]
    cost_matrix = -cost_matrix

    # 匈牙利算法
    rows, cols = cost_matrix.shape
    star = np.zeros((rows, cols), dtype=np.int8)
    prime = np.zeros((rows, cols), dtype=np.int8)
    covered_rows = np.zeros(rows, dtype=np.int8)
    covered_cols = np.zeros(cols, dtype=np.int8)
    while not np.all(covered_rows) or not np.all(covered_cols):
        for i in range(rows):
            if not covered_rows[i]:
                j = int(cv2.minMaxLoc(-cost_matrix[i, :])[1])
                if not covered_cols[j]:
                    star[i][j] = 1
                    covered_rows[i] = 1
                    covered_cols[j] = 1
                    break
        for j in range(cols):
            if not covered_cols[j]:
                for i in range(rows):
                    if not covered_rows[i] and star[i][j] == 0:
                        if cost_matrix[i][j] < cost_matrix[i][cols] + cost_matrix[rows][j] - cost_matrix[rows][cols]:
                            cost_matrix[i][j] = cost_matrix[i][cols] + cost_matrix[rows][j] - cost_matrix[rows][cols]
                break
        for i in range(rows):
            if not covered_rows[i]:
                for j in range(cols):
                    if not covered_cols[j] and star[i][j] == 0:
                        if cost_matrix[i][j] == cost_matrix[i][cols] + cost_matrix[rows][j] - cost_matrix[rows][cols]:
                            star[i][j] = 1
                            covered_rows[i] = 1
                            covered_cols[j] = 1
                            break
        for j in range(cols):
            if not covered_cols[j]:
                for i in range(rows):
                    if not covered_rows[i] and star[i][j] == 1:
                        prime[i][j] = 1
                        covered_rows[i] = 1
                        covered_cols[j] = 1
                        break

    matched_indices = []
    for i in range(rows):
        for j in range(cols):
            if star[i][j] == 1 and prime[i][j] == 1:
                matched_indices.append((i, j))

    return matched_indices


# 示例数据
masks = [(100, 150, 50, 50), (200, 250, 50, 50)]  # 佩戴口罩的边界框
nomask = [(150, 200, 50, 50), (250, 300, 50, 50)]  # 未佩戴口罩的边界框
pedestrians = [(75, 100, 100, 200), (175, 250, 100, 200)]  # 行人边界框

# 执行匹配
matched_pedestrians = match_masks_with_pedestrians(masks, nomask, pedestrians)
print(matched_pedestrians)