from time import time, sleep
from threading import Thread

def wait(element):
    print(f"Making element-{element} wait for 10 secs\n")
    sleep(10)
    
if __name__ == '__main__':
    print("Begin ...\n")
    begin = time()
    threads = []
    for element in ['A','B','C']:
        thread = Thread(target = wait,
                        args = (element,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    print("Total execution time is: ", time() - begin)
