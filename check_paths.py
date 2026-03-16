import os

# 您脚本里写的路径
paths_to_check = [
    "./checkpoints_viz/baseline/model.pt",
    "./checkpoints_viz/bio_gradient/model.pt"
]

print(f"1. 当前工作目录 (os.getcwd): {os.getcwd()}")
print("-" * 40)

for p in paths_to_check:
    abs_path = os.path.abspath(p)
    exists = os.path.exists(p)
    print(f"检查路径: {p}")
    print(f" -> 绝对路径: {abs_path}")
    print(f" -> 是否存在: {'✅ 存在' if exists else '❌ 不存在'}")
    if not exists:
        # 尝试列出上级目录看看有什么
        parent = os.path.dirname(p)
        if os.path.exists(parent):
            print(f"    (上级目录 {parent} 存在，包含文件: {os.listdir(parent)})")
        else:
            print(f"    (上级目录 {parent} 也不存在)")
    print("-" * 40)