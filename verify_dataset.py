import os

for split in ("train", "val"):
    root = os.path.join("data", split)
    print(f"\n{split.upper()}:")
    if not os.path.exists(root):
        print("  (folder missing)")
        continue
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    total = 0
    for c in classes:
        n = len([f for f in os.listdir(os.path.join(root, c)) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        total += n
        print(f"  {c}: {n}")
    print(f"  Total {split}: {total}")
