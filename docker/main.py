from customer_headroom.config import load_config
from dtaml.logging import get_logger
from time import sleep

get_logger("test_docker")
print(load_config("dev").dumps())
print("test")

while True:
    sleep(10)
