from multiprocessing import Process, freeze_support
import pymp

def f():
    print ('hello world!')
    ex_array = pymp.shared.array((100,), dtype='uint8')
    # with pymp.Parallel(4) as p:
    #     for index in p.range(0, 100):
    #         ex_array[index] = 1
    #         print(index)
    #         # The parallel print function takes care of asynchronous output.
    # #         p.print('Yay! {} done!'.format(index))

if __name__ == '__main__':
    freeze_support()
    Process(target=f).start()

