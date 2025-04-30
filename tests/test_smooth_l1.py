from exercises.smooth_l1 import smooth_l1
import numpy as np


def test_smooth_l1():
    # sigma = 1.0 对应 beta = 1.0
    sigma = 1.0

    # 测试样例1：误差较小
    y_pred1 = np.array([0.1, 1.2, -0.3])
    y_true1 = np.array([0.0, 1.0, 0.0])
    x1 = y_pred1 - y_true1  # 计算差值
    expected_losses1 = np.array(
        [0.5 * (sigma * 0.1) ** 2, 0.5 * (sigma * 0.2) ** 2, 0.5 * (sigma * -0.3) ** 2]
    )
    expected1 = np.mean(expected_losses1)
    # 修正函数调用和参数
    result_losses1 = smooth_l1(x1, sigma=sigma)
    result1 = np.mean(result_losses1)
    print(
        f"测试样例1 (小误差): Expected={expected1:.4f}, Got={result1:.4f} ->",
        "yes" if np.allclose(result1, expected1) else "no",
    )

    # 测试样例2：误差较大
    y_pred2 = np.array([2.5, -1.8, 0.5])
    y_true2 = np.array([0.0, 0.0, 2.0])
    x2 = y_pred2 - y_true2  # 计算差值
    expected_losses2 = np.array(
        [
            abs(2.5) - 0.5 / sigma**2,
            abs(-1.8) - 0.5 / sigma**2,
            abs(-1.5) - 0.5 / sigma**2,
        ]
    )
    expected2 = np.mean(expected_losses2)
    # 修正函数调用和参数
    result_losses2 = smooth_l1(x2, sigma=sigma)
    result2 = np.mean(result_losses2)
    print(
        f"测试样例2 (大误差): Expected={expected2:.4f}, Got={result2:.4f} ->",
        "yes" if np.allclose(result2, expected2) else "no",
    )

    # 测试样例3：混合误差
    y_pred3 = np.array([0.2, 1.0, -2.0, 3.0])
    y_true3 = np.array([0.0, 0.5, 0.0, 1.5])
    x3 = y_pred3 - y_true3  # 计算差值
    expected_losses3 = np.array(
        [
            0.5 * (sigma * 0.2) ** 2,
            0.5 * (sigma * 0.5) ** 2,
            abs(-2.0) - 0.5 / sigma**2,
            abs(1.5) - 0.5 / sigma**2,
        ]
    )
    expected3 = np.mean(expected_losses3)
    # 修正函数调用和参数
    result_losses3 = smooth_l1(x3, sigma=sigma)
    result3 = np.mean(result_losses3)
    print(
        f"测试样例3 (混合误差): Expected={expected3:.4f}, Got={result3:.4f} ->",
        "yes" if np.allclose(result3, expected3) else "no",
    )


if __name__ == "__main__":
    test_smooth_l1()
