import yaml
import sys

with open(sys.argv[1], 'r') as file:
    arguments = yaml.safe_load(file)

print(arguments['model_name'])
