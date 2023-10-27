from pynput.mouse import Button, Controller
import time
from datetime import datetime
mouse = Controller()
time_start = time.time()
hours = 2
minutes = 10

duration = (hours * 60  + minutes) * 60
timestamp = time.time()  # Get the current timestamp
# Convert the timestamp to a datetime object
dt_object = datetime.fromtimestamp(timestamp)

# Format the datetime object as a string
formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

print(f'Schedule clicking for {hours} h {minutes} min')
print("START time:", formatted_time)

while time.time() - timestamp < duration:
    mouse.click(Button.left, 1)
    time.sleep(100)

print('END CLICKING')

timestamp = time.time()  # Get the current timestamp
# Convert the timestamp to a datetime object
dt_object = datetime.fromtimestamp(timestamp)

# Format the datetime object as a string
formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

print("Current time:", formatted_time)
