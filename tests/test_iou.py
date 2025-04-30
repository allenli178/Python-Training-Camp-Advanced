from exercises.iou import calculate_iou


def test_cases():
    # 测试样例1：完全重叠的情况
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    result1 = calculate_iou(box1, box2)
    print(f"测试样例1 - IoU: {result1:.4f}")
    print("样例1:", "yes" if abs(result1 - 1.0) < 1e-5 else "no")

    # 测试样例2：部分重叠的情况
    box3 = [5, 5, 15, 15]
    box4 = [10, 10, 20, 20]
    result2 = calculate_iou(box3, box4)
    print(f"测试样例2 - IoU: {result2:.4f}")
    print("样例2:", "yes" if abs(result2 - 0.14285714) < 1e-5 else "no")

    # 测试样例3：无重叠的情况
    box5 = [0, 0, 10, 10]
    box6 = [20, 20, 30, 30]
    result3 = calculate_iou(box5, box6)
    print(f"测试样例3 - IoU: {result3:.4f}")
    print("样例3:", "yes" if result3 == 0 else "no")


if __name__ == "__main__":
    test_cases()
