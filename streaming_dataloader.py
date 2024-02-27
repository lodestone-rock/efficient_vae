import time
import random
import logging
from queue import Queue
from typing import Optional, List
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import requests
import cv2
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR)


def threaded_data_loader(
        dataset, 
        batch_size=1, 
        num_workers=10, 
        collate_fn=None,
        shuffle=False, 
        prefetch_factor=4, 
        seed=0, 
        worker_timeout=None, 
        dataloader_timeout=60, 
        debug=False
    ):
    random.seed(seed)
    # Create a ThreadPoolExecutor with the specified number of workers
    executor = ThreadPoolExecutor(max_workers=num_workers)
    
    # Define a function to fetch and process a batch of samples       
    def process_batch(idx):
        return dataset[idx]


    # Generate batches of indices based on the dataset size and batch size
    num_samples = len(dataset)
    if shuffle:
        indices = list(range(num_samples))
        random.shuffle(indices)
        batch_indices = [indices[i:i+batch_size] for i in range(0, num_samples, batch_size)]
    else:
        batch_indices = [list(range(i, min(i + batch_size, num_samples))) for i in range(0, num_samples, batch_size)]

    # Create a prefetch queue to store prefetched batches
    prefetch_queue = Queue(maxsize=prefetch_factor)

        # Function to prefetch batches of samples
    def prefetch_batches():
        for indices in batch_indices:
            tasks = [executor.submit(process_batch, batch) for batch in indices]
            samples = []
            for task in tasks:
                try:
                    samples.append(task.result(timeout=worker_timeout))
                except Exception as e:
                    if debug:
                        print(f"stale batch at index {indices}")
                    samples.append(None)
            if len(samples) > 0:
                prefetch_queue.put(samples)
            else:
                prefetch_queue.put("dummy")

    # Function to yield batches of samples
    def data_generator():
        current_index = 0
        while True:
            prefetch_thread = executor.submit(prefetch_batches)
            next_batch = prefetch_queue.get(timeout=dataloader_timeout)
            if collate_fn:
                yield collate_fn(next_batch)
            else:
                yield next_batch
            current_index += 1
            if current_index == len(batch_indices):
                executor.shutdown()
                break

    # Define the data loader generator function
    def data_loader():
        for batch in data_generator():
            yield batch

    return data_loader

def random_crop(image, crop_size):
    h, w = image.shape[:2]
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    cropped_image = image[top:top+crop_size, left:left+crop_size]
    return cropped_image

def rotate(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def scale(image, scale_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image


class CustomDataset():
    # just streams image from internet, and return either none or augmented image
    def __init__(self, csv_file, square_size=50, seed=84):

        self.data = pd.read_parquet(csv_file)
        self.square_size = square_size
        random.seed(seed)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        try:
            print(idx)
            response = requests.get(row['url'])
            img = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            
            # img = scale(image, random.random() * 0.4 + 0.8) # bad idea tbh
            img = random_crop(img, self.square_size)
            img = rotate(img, random.choice([0, 90, 180, 270]))

            # img = np.transpose(img, (2, 0, 1))
            # img = torch.from_numpy(img).float()
            
            return img
        except Exception as e:
            logging.error(f"Error loading image at index {idx}: {str(e)}")
            return None
        

# collate function here is to fill the gap due to lossy retrieval by simply filling the gap with random rotation of random sample

def collate_fn(batch):
    # count none in batch and drop the batch if it's exceeded the treshold value
    none_count = 0
    print(none_count)
    for x in batch:
        if x is None:
            none_count += 1
    # if none_count > len(batch)//2:
    #     return None
    # else:
    truncated_batch = [arr for arr in batch if arr is not None]

    # Replace the None values with random existing arrays
    truncated_batch = [rotate(random.choice(truncated_batch), random.choice([0, 90, 180, 270])) for arr in range(none_count)] + truncated_batch

    # Concatenate the filled arrays
    result = np.stack(truncated_batch, axis=0)

    return result


csv_file = "0035af9f90f581816acf269df5eb37ad.parquet"
dataset = CustomDataset(csv_file, square_size=256)



t_dl = threaded_data_loader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,  num_workers=50, prefetch_factor=100, worker_timeout=5)

data = []
for i, x in enumerate(t_dl()):
    if i > 3:
        break
    data.append(x)
print()
