import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
base_dir1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir2 = os.path.dirname(os.path.abspath(__file__))
base_dir3 = os.path.abspath(__file__)

config = {
    'logs': 'logs/'
}

datasets_path = {
    'air': 'datasets/air_quality'
}

epsilon = 1e-8

for k, v in datasets_path.items():
    datasets_path[k] = os.path.join(base_dir, v)

if __name__ == '__main__':
    print(datasets_path)
#  print(base_dir1)
#  print(base_dir2)
#  print(base_dir3)
