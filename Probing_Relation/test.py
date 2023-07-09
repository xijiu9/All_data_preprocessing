import multiprocessing as mp
import time

def worker(shared_dict, i):
    value = i**2
    print(f"Adding {i}: {value}")
    shared_dict[i] = value
    time.sleep(1)
    print(f"Done adding {i}")

if __name__ == '__main__':
    manager = mp.Manager()
    shared_dict = manager.dict()

    print(f"Empty dictionary: {shared_dict.items()}")

    processes = []
    for i in range(10):
        p = mp.Process(target=worker, args=(shared_dict, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Final dictionary: {shared_dict.items()}")
