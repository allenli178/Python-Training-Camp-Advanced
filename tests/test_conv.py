from exercises.conv import conv2d
import numpy as np


def test_convolution():
    # 测试样例1：全零测试
    img1 = np.zeros((5, 5))
    kernel1 = np.random.rand(3, 3)
    result1 = conv2d(img1, kernel1)
    print(
        "测试样例1（全零输入）:",
        "yes" if np.allclose(result1, np.zeros((3, 3))) else "no",
    )

    # 测试样例2：对角核测试
    img2 = np.eye(5)
    kernel2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result2 = conv2d(img2, kernel2)
    expected2 = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    print("测试样例2（对角检测）:", "yes" if np.allclose(result2, expected2) else "no")

    # 测试样例3：求和核测试
    img3 = np.arange(25).reshape(5, 5)
    kernel3 = np.ones((3, 3))
    result3 = conv2d(img3, kernel3)
    expected3 = np.array([[54, 63, 72], [99, 108, 117], [144, 153, 162]])
    print("测试样例3（区域求和）:", "yes" if np.allclose(result3, expected3) else "no")


if __name__ == "__main__":
    test_convolution()
