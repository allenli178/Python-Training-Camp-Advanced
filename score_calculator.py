import re
import subprocess
import sys
import os
from collections import defaultdict


def run_pytest_and_score():
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # 设置环境变量PYTHONPATH，使测试进程也能找到模块
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = project_root + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = project_root

    # 运行pytest并捕获输出
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    # 解析测试结果
    test_results = defaultdict(dict)
    current_file = None

    # 使用正则匹配测试结果行
    pattern = re.compile(r"tests[\\/](test_\w+)\.py::([\w:]+) (\w+)")

    for line in result.stdout.split("\n"):
        match = pattern.search(line)
        if match:
            file_name, test_name, status = match.groups()
            test_results[file_name][test_name] = status == "PASSED"

    # 计算得分
    total_tests = sum(len(tests) for tests in test_results.values())
    passed_tests = sum(
        sum(1 for passed in tests.values() if passed) for tests in test_results.values()
    )

    score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    # 创建进度条
    progress_bar_width = 50  # 进度条宽度
    progress_percentage = passed_tests / total_tests if total_tests > 0 else 0
    completed_blocks = int(progress_percentage * progress_bar_width)
    progress_bar = "#" * completed_blocks + "-" * (
        progress_bar_width - completed_blocks
    )

    # 打印详细报告
    print("\n" + "=" * 50)
    print("测试结果详情:")
    for file_name, tests in test_results.items():
        print(f"\n{file_name}:")
        for test_name, passed in tests.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {test_name:50} {status}")

    print("\n" + "=" * 50)
    print(f"进度条: [{progress_bar}] {progress_percentage:.1%}")
    print(f"最终得分: {score:.1f}分 (通过 {passed_tests}/{total_tests} 个测试)")
    print("=" * 50 + "\n")

    # 保存结果到文件
    with open("test_score.txt", "w") as f:
        f.write(f"Passed: {passed_tests}/{total_tests}\n")
        f.write(f"Score: {score:.1f}\n")
        f.write(f"Progress: [{progress_bar}]\n")

    return score


if __name__ == "__main__":
    run_pytest_and_score()
