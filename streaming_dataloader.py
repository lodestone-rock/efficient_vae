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
# logging.basicConfig(filename='dataset_errors.log', level=logging.ERROR)

from concurrent.futures import ThreadPoolExecutor
import random
from queue import Queue

def threading_dataloader(dataset, batch_size=1, num_workers=10, collate_fn=None, shuffle=False, prefetch_factor=4, seed=0, timeout=None):
    """
    A function to load data using multiple threads. This function can be used to speed up the data loading process. 
    
    Parameters:
    dataset (iterable): The dataset to load.
    batch_size (int, optional): The number of samples per batch. Defaults to 1.
    num_workers (int, optional): The number of worker threads to use. Defaults to 10.
    collate_fn (callable, optional): A function to collate samples into a batch. If None, the default collate_fn is used.
    shuffle (bool, optional): Whether to shuffle the dataset before loading. Defaults to False.
    prefetch_factor (int, optional): The number of batches to prefetch. Defaults to 4.
    seed (int, optional): The seed for the random number generator. Defaults to 0.
    timeout (int, optional): The maximum number of seconds to wait for a batch. If None, there is no timeout.

    Yields:
    object: A batch of data.
    """
    # Initialize a random number generator with the given seed
    random.seed(seed)

    # Create a ThreadPoolExecutor with the specified number of workers
    workers = ThreadPoolExecutor(max_workers=num_workers)
    overseer = ThreadPoolExecutor(max_workers=1)

    # Generate batches of indices based on the dataset size and batch size
    num_samples = len(dataset)
    batch_indices = [list(range(i, min(i + batch_size, num_samples))) for i in range(0, num_samples, batch_size)]
    if shuffle:
        indices = list(range(num_samples))
        random.shuffle(indices)
        batch_indices = [indices[i:i+batch_size] for i in range(0, num_samples, batch_size)]

    # Create a queue to store prefetched batches
    prefetch_queue = Queue(maxsize=prefetch_factor * num_workers)
    
    # Function to prefetch batches of samples
    def batch_to_queue(indices):
        """
        Function to load a batch of data and put it into the prefetch queue.

        Parameters:
        indices (list): The indices of the samples in the batch.
        """
        batch = [dataset[i] for i in indices]
        if collate_fn is not None:
            batch = collate_fn(batch)
        prefetch_queue.put(batch) # 1. if you want to ensure order use return instead of queue here
    
    # Submit the prefetch tasks to the worker threads
    def overseer_thread():
        for indices in batch_indices:
            workers.submit(batch_to_queue, indices) # 2. then store the future and re itterate it here and get the value

    # just in case worker submit loop is too slow due to a ton of loops, fork it to another thread so the main thread can continue
    overseer.submit(overseer_thread)

    # Yield the prefetched batches
    for _ in range(len(batch_indices)):
        yield prefetch_queue.get(timeout=timeout)


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
            # print(idx)
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
            return self.__getitem__(random.randrange(0, len(self.data)))
            # logging.error(f"Error loading image at index {idx}: {str(e)}")
            return None
        

# collate function here is to fill the gap due to lossy retrieval by simply filling the gap with random rotation of random sample
def collate_fn(batch):
    # count none in batch and drop the batch if it's exceeded the treshold value
    none_count = 0
    # print(none_count)
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

