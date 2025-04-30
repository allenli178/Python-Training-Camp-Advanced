from exercises.maxpool import maxpool
import numpy as np


def test_maxpool():
    # 测试样例1：标准 2x2 pooling, stride 2
    input1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    pool_size1 = (2, 2)
    stride1 = (2, 2)
    expected1 = np.array([[6, 8], [14, 16]])
    result1 = maxpool(input1, kernel_size=pool_size1[0], stride=stride1[0])
    print(
        "测试样例1 (2x2, stride 2):", "yes" if np.allclose(result1, expected1) else "no"
    )

    # 测试样例2：3x3 pooling, stride 1
    input2 = np.array([[1, 3, 2, 4], [5, 0, 7, 1], [6, 8, 2, 9], [2, 5, 1, 3]])
    pool_size2 = (3, 3)
    stride2 = (1, 1)
    expected2 = np.array([[8, 9], [8, 9]])
    result2 = maxpool(input2, kernel_size=pool_size2[0], stride=stride2[0])
    print(
        "测试样例2 (3x3, stride 1):", "yes" if np.allclose(result2, expected2) else "no"
    )

    # 测试样例3：包含负数的 2x2 pooling
    input3 = np.array([[-1, -5, 2], [-8, 0, -3], [4, -2, 1]])
    pool_size3 = (2, 2)
    stride3 = (1, 1)
    expected3 = np.array([[0, 2], [4, 1]])
    result3 = maxpool(input3, kernel_size=pool_size3[0], stride=stride3[0])
    print(
        "测试样例3 (2x2, stride 1, neg):",
        "yes" if np.allclose(result3, expected3) else "no",
    )


if __name__ == "__main__":
    test_maxpool()
