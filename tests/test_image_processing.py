import cv2
import numpy as np
import pytest
import os
from exercises.image_processing import image_processing_pipeline


def test_image_processing_pipeline():
    # 使用相对路径访问测试图片
    image_path = os.path.join("picture", "1.png")

    # 检查图片是否存在
    if not os.path.exists(image_path):
        pytest.skip(f"Test image not found at {image_path}")

    # 执行处理流程
    edges = image_processing_pipeline(image_path)

    # 验证结果
    assert edges is not None, "边缘检测失败"
    assert isinstance(edges, np.ndarray), "返回值不是NumPy数组"
    assert edges.ndim == 2, "边缘图像应该是二维灰度图"

    # 验证图像尺寸
    original = cv2.imread(image_path)
    assert edges.shape == original.shape[:2], "边缘图像尺寸与原始图像不匹配"


def test_invalid_image_path():
    # 测试无效路径
    edges = image_processing_pipeline("invalid_path.png")
    assert edges is None, "对无效路径应该返回None"
