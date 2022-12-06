'''
Cifar-10
'''
#python main.py --model fedavg --device_id 7 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model fedprox --device_id 6 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model moon --device_id 5 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model fedavgnorm --device_id 7 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model fedournormlogexp --device_id 6 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model fedproc --device_id 5 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model feddyn --device_id 7 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model fedopt --device_id 6 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model feddyn --device_id 4 --dataset fl_cifar10 --beta 0.1 --csv_log &
#python main.py --model fedopt --device_id 4 --dataset fl_cifar10 --beta 0.1 --csv_log &

#python main.py --model fedproc --device_id 4 --dataset fl_cifar10 --beta 0.5 --csv_log &
#python main.py --model fedproc --device_id 4 --dataset fl_cifar10 --beta 0.5 --csv_log &

#python main.py --model fedrs --device_id 6 --dataset fl_cifar10 --beta 0.3 --csv_log &
#python main.py --model fedrs --device_id 6 --dataset fl_cifar10 --beta 0.1 --csv_log &



'''
Cifar-100
'''
python main.py --model fedavg --device_id 7 --dataset fl_cifar100 --beta 0.5 --csv_log &
python main.py --model fedprox --device_id 6 --dataset fl_cifar100 --beta 0.5 --csv_log &
python main.py --model moon --device_id 5 --dataset fl_cifar100 --beta 0.5 --csv_log &
python main.py --model fedproc --device_id 4 --dataset fl_cifar100 --beta 0.5 --csv_log &
python main.py --model fedavgnorm --device_id 4 --dataset fl_cifar100 --beta 0.5 --csv_log