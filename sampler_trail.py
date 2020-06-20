from multiprocessing import Process, Queue
import numpy as np


def sample_function(batch_size, result_queue, SEED):
    np.random.seed(SEED)
    one_batch = []  # consisting of tuples from sample, with a length of batch size.
    for i in range(batch_size):
        one_batch.append(np.random.randint(1, 100))
    result_queue.put(one_batch)


processors = []
result_queue = Queue(maxsize=10)
for i in range(3):
    processors.append(Process(target=sample_function, args=(10, result_queue, np.random.randint(2e9))))
    processors[i].start()


for process in processors:
    process.join()

while not result_queue.empty():
    print(result_queue.get())
