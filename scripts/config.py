import os

project_root = "Amazon Unstop"

train_csv = os.path.join(project_root, "dataset", "train.csv")
test_csv = os.path.join(project_root, "dataset", "test.csv")
train_image_dir = os.path.join(project_root, "images", "train")
test_image_dir = os.path.join(project_root, "images", "test")
cache_dir = os.path.join(project_root, "cache")
output_csv = os.path.join(project_root, "test_out.csv")

# Other configurations
batch_size = 32
image_size = (224, 224)