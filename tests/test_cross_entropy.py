from exercises.cross_entropy import cross_entropy_loss
import numpy as np


def test_cross_entropy():
    # 测试样例1：二分类，高置信度预测
    y_pred1 = np.array([[0.1, 0.9], [0.8, 0.2]])
    y_true1 = np.array([1, 0])
    expected1 = -(np.log(0.9) + np.log(0.8)) / 2
    result1 = cross_entropy_loss(y_true1, y_pred1)
    print(
        f"测试样例1 (二分类, 高置信度): Expected={expected1:.4f}, Got={result1:.4f} ->",
        "yes" if np.allclose(result1, expected1) else "no",
    )

    # 测试样例2：三分类，较低置信度预测
    y_pred2 = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    y_true2 = np.array([1, 0])
    expected2 = -(np.log(0.5) + np.log(0.6)) / 2
    result2 = cross_entropy_loss(y_true2, y_pred2)
    print(
        f"测试样例2 (三分类, 低置信度): Expected={expected2:.4f}, Got={result2:.4f} ->",
        "yes" if np.allclose(result2, expected2) else "no",
    )

    # 测试样例3：多分类，不同批大小
    y_pred3 = np.array(
        [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]]
    )
    y_true3 = np.array([0, 1, 3])
    expected3 = -(np.log(0.7) + np.log(0.7) + np.log(0.1)) / 3
    result3 = cross_entropy_loss(y_true3, y_pred3)
    print(
        f"测试样例3 (四分类, 批大小3): Expected={expected3:.4f}, Got={result3:.4f} ->",
        "yes" if np.allclose(result3, expected3) else "no",
    )


if __name__ == "__main__":
    test_cross_entropy()
