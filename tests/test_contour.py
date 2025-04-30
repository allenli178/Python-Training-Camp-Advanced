import pytest
import cv2
import numpy as np
import os
from exercises.contour_detection import contour_detection


class TestContourDetection:
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.test_img = os.path.join("picture", "7.png")
        if not os.path.exists(cls.test_img):
            pytest.skip(f"测试图片 {cls.test_img} 不存在")

    def test_success_case(self):
        """测试正常图片处理"""
        result, contours = contour_detection(self.test_img)

        assert result is not None, "结果图像不应为None"
        assert isinstance(contours, list), "轮廓应返回列表"
        assert len(contours) > 0, "应检测到至少一个轮廓"

        # 验证结果图像格式
        assert isinstance(result, np.ndarray), "结果应为NumPy数组"
        assert result.ndim == 3, "结果图像应为3通道彩色图"

    def test_invalid_path(self):
        """测试无效路径"""
        result, contours = contour_detection("invalid_path.jpg")
        assert result is None, "无效路径应返回None"
        assert contours is None, "无效路径应返回None"
