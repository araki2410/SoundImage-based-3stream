import argparse
import time

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m',
        default='resnet',
        type=str,
        help='[ vgg16 | resnet ]')
    parser.add_argument(
        '--fps', '-f',
        default=6,
        type=int,
        help='1 to 30')
    parser.add_argument(
        '--annotation_file', '-a',
        default='./Annotate/73',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--optimizer', '-o',
        default="adam",
        type=str,
        help='[ adam | sgd ]')
    parser.add_argument(
        '--lr',
        default=0.000001,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--step_rate',
        default=1.0,
        type=float,
        help='learning rate * down_rate. default=1.0 (no swing)')
    parser.add_argument(
        '--step_period',
        default=1000,
        type=int,
        help='piriod of down learning_rate. default=100')
    parser.add_argument(
        '--gpu', '-g', default=0, type=int,
        help='0..9: single gpu | -1: multi_gpu')
    parser.add_argument(
        '--epochs', '-e',
        default=50,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--batch_size', '-b', default=32, type=int, help='Batch Size')
    parser.add_argument(
        '--num_works', '-n', default=8, type=int, help='Number of works')
    parser.add_argument(
        '--threthold', default=0.5, type=float, help='threthold for N-ok-K')
    parser.add_argument(
        '--stream',
        default="rgb",
        type=str,
        help='Core name to save for: image, logs and other outputs')
    parser.add_argument(
        '--jpg_path', '-j',
        default='./Dataset',
        type=str,
        help='Directory path of Datas')
    parser.add_argument(
        '--result_path',
        default='./Outputs',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--image_path',
        default='./Outputs',
        type=str,
        help='output image path')
    parser.add_argument(
        '--oldloss',
        default=2,
        type=int,
        help='The first loss score. If test_loss less than oldloss, save model parameta.')
    parser.add_argument(
        '--multi_gpu', 
        help='If true use mulch GPU.## This is an old option') 

    parser.add_argument('--start_time', type=str)
    start_time = time.localtime()
    parser.set_defaults(start_time="{year}{month:02}{day:02}-{hour:02}{minute:02}".format(year=start_time.tm_year,
                                                                                          month=start_time.tm_mon,
                                                                                          day=start_time.tm_mday,
                                                                                          hour=start_time.tm_hour,
                                                                                          minute=start_time.tm_min))
    parser.set_defaults(mulch_gpu=False)


    args = parser.parse_args()

    return args
