import os
import time
import threading
import argparse
import tensorflow as tf


def thread_main(path):

    def write_data(coord, path):
        while not coord.should_stop():
            with open(path, 'r+') as f:
                if len(f.readlines()) >= 100:
                    coord.request_stop()
                else:
                    f.write('1\n')

    coord = tf.train.Coordinator()
    threads = [threading.Thread(target=write_data, args=(coord, path)) for i in range(10)]

    for t in threads:
        t.start()
    coord.join(threads)


def no_thread_main(path):
    do_stop = False
    while not do_stop:
        with open(path, 'r+') as f:
            if len(f.readlines()) >= 100:
                do_stop = True
            else:
                f.write('1\n')


def main(is_thread=True):
    start = time.time()
    filename = 'coordinator.txt'
    target_path = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(target_path):
        with open(target_path, 'w') as f:
            f.write('')

    if is_thread:
        thread_main(target_path)
    else:
        no_thread_main(target_path)

    end = 'time: {:.4f} sec'.format(time.time() - start)
    print(end)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--thread', action='store_true', default=False)
    options = args.parse_args()

    main(options.thread)
