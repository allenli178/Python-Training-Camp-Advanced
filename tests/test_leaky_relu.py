from exercises.leaky_relu import leaky_relu
import numpy as np


def test_leaky_relu():
    # 测试样例1：混合输入，默认 alpha=0.01
    input1 = np.array([-3.0, -1.5, 0.0, 2.1, 5.0])
    expected1 = np.array([-0.03, -0.015, 0.0, 2.1, 5.0])
    result1 = leaky_relu(input1)
    print(
        "测试样例1（混合输入, alpha=0.01）:",
        "yes" if np.allclose(result1, expected1) else "no",
    )

    # 测试样例2：全负输入，指定 alpha=0.2
    input2 = np.array([-10.0, -5.5, -0.1])
    alpha2 = 0.2
    expected2 = np.array([-2.0, -1.1, -0.02])
    result2 = leaky_relu(input2, alpha=alpha2)
    print(
        "测试样例2（全负输入, alpha=0.2）:",
        "yes" if np.allclose(result2, expected2) else "no",
    )

    # 测试样例3：非负输入，任意 alpha
    input3 = np.array([0.0, 1.1, 100.0, 42.0])
    alpha3 = 0.5
    expected3 = np.array([0.0, 1.1, 100.0, 42.0])
    result3 = leaky_relu(input3, alpha=alpha3)
    print(
        "测试样例3（非负输入, alpha=0.5）:",
        "yes" if np.allclose(result3, expected3) else "no",
    )


if __name__ == "__main__":
    test_leaky_relu()
