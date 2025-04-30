from exercises.nms import nms


def test_cases():
    # 测试样例1：多个重叠框
    boxes2 = [
        [10, 10, 60, 60],  # 得分最高
        [15, 15, 65, 65],  # 与第一个框IoU≈0.53
        [50, 50, 100, 100],  # 与第一个框IoU≈0.11
        [150, 150, 200, 200],  # 完全独立
    ]
    scores2 = [0.9, 0.85, 0.8, 0.75]
    result2 = nms(boxes2, scores2, 0.5)
    expected2 = [0, 2, 3]
    print("测试样例2:", "yes" if set(result2) == set(expected2) else "no")

    # 测试样例2：完全不重叠的多个框
    boxes3 = [[0, 0, 30, 30], [40, 40, 70, 70], [80, 80, 110, 110]]
    scores3 = [0.8, 0.7, 0.6]
    result3 = nms(boxes3, scores3, 0.5)
    expected3 = [0, 1, 2]
    print("测试样例3:", "yes" if set(result3) == set(expected3) else "no")

    # 测试样例3：完全相同位置的框
    boxes4 = [[0, 0, 50, 50], [0, 0, 50, 50], [0, 0, 50, 50]]
    scores4 = [0.9, 0.8, 0.7]
    result4 = nms(boxes4, scores4, 0.5)
    expected4 = [0]
    print("测试样例4:", "yes" if result4 == expected4 else "no")

    # 测试样例4：空输入
    boxes5 = []
    scores5 = []
    result5 = nms(boxes5, scores5, 0.5)
    expected5 = []
    print("测试样例5:", "yes" if result5 == expected5 else "no")


if __name__ == "__main__":
    test_cases()
