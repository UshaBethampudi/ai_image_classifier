from PIL import Image
import os

def verify_images(folder):
    bad_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                img = Image.open(path)
                img.verify()  # verify image integrity
            except Exception as e:
                bad_files.append(path)
                print(f"❌ Bad file: {path} ({e})")
    return bad_files

if __name__ == "__main__":
    bad = verify_images("data/train")
    bad += verify_images("data/val")
    print("\nTotal bad files:", len(bad))
