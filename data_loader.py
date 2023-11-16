# 1. opencv >>>> PIL
#
# IO cost
# 1. hdf5, tfrecord (Tensorflow)  several small image (text) to a larger file
# 2. preloader
# 3. cache (lmdb or redis) key, value cahe
#
# pin_memory = True

# tfrecord
import tensorflow as tf
import os
from PIL import Image
import numpy as np

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_path, label):
    image_raw = Image.open(image_path)
    image_raw = np.array(image_raw)

    feature = {
        'image': _bytes_feature(image_raw.tobytes()),
        'label': _int64_feature(label),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_tfrecord(output_filename, image_folder):
    writer = tf.io.TFRecordWriter(output_filename)

    # Your dataset: list of tuples (image_path, label)
    dataset = [
        ("path/to/image1.jpg", 0),
        ("path/to/image2.jpg", 1),
        # Add more images and labels as needed
    ]

    for image_path, label in dataset:
        tf_example = image_example(image_path, label)
        writer.write(tf_example.SerializeToString())

    writer.close()

# Specify the output TFRecord file and the folder containing images
tfrecord_filename = 'output.tfrecord'
image_folder_path = 'path/to/your/image/folder'

# Create TFRecord file
create_tfrecord(tfrecord_filename, image_folder_path)

# several small images to hdf5 file
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
from PIL import Image
import os
from multiprocessing import Pool

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(".jpg")]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

def process_images(args):
    idx, image_path = args
    image = Image.open(image_path).convert('RGB')
    image = transforms.ToTensor()(image)
    return idx, image.numpy()

def convert_to_hdf5_multiprocess(dataset, output_filename='output.h5', num_processes=4):
    with h5py.File(output_filename, 'w') as hdf5_file:
        images_group = hdf5_file.create_group('images')

        # Use multiprocessing to parallelize image processing
        with Pool(num_processes) as pool:
            results = pool.map(process_images, enumerate(dataset.image_paths))

        for idx, image_data in results:
            images_group.create_dataset(f'image_{idx}', data=image_data)

# Specify the folder containing images
image_folder_path = 'path/to/your/image/folder'

# Create a dataset
dataset = CustomDataset(image_folder_path)

# Convert to HDF5 using multiple processes
hdf5_filename = 'output.h5'
convert_to_hdf5_multiprocess(dataset, hdf5_filename, num_processes=4)

# redis 
import redis
from PIL import Image
from io import BytesIO
import base64

# Connect to the Redis server
redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

def decode_image(encoded_image):
    decoded_image = base64.b64decode(encoded_image)
    return Image.open(BytesIO(decoded_image))

def store_images_in_redis(image_folder):
    # Assuming each image has a unique filename
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        encoded_image = encode_image(image_path)
        key = f"image:{image_file}"
        redis_client.set(key, encoded_image)

def retrieve_image_from_redis(image_key, output_path):
    encoded_image = redis_client.get(image_key)
    if encoded_image:
        decoded_image = decode_image(encoded_image)
        decoded_image.save(output_path)

# Specify the folder containing images
image_folder_path = 'path/to/your/image/folder'

# Store images in Redis
store_images_in_redis(image_folder_path)

# Retrieve an image from Redis (replace 'your_image_key' with an actual key)
image_key = 'image:your_image.jpg'
output_image_path = 'output_image.jpg'
retrieve_image_from_redis(image_key, output_image_path)


