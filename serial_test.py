import datetime

import pigpio
import time

def normalize(things):
    dis = ''
    for distance in things:
        if distance >= 36:
            dis += '8'
        elif distance >= 24:
            dis += '7'
        elif distance >= 20:
            dis += '6'
        elif distance >= 16:
            dis += '5'
        elif distance >= 12:
            dis += '4'
        elif distance >= 10:
            dis += '3'
        elif distance >= 8:
            dis += '2'
        elif distance >= 6:
            dis += '1'
        else:
            dis += '0'
    return dis

time_record = int(time.time() * 1000)
time_limit = 50
pi = pigpio.pi()
sensor_message_size = 3
sensor_signal_pin = 4
h1 = pi.serial_open("/dev/ttyAMA0")
while True:
    while (int(time.time() * 1000) - time_record) >= time_limit:
        time.sleep(0.002)
    time_record = int(time.time() * 1000)
    distance = []
    pi.serial_read(h1)  # clear any redauntancy data
    pi.write(sensor_signal_pin, pigpio.HIGH)
    while pi.serial_data_available(h1) < sensor_message_size:
        time.sleep(0.0007)
    (b, d) = pi.serial_read(h1, sensor_message_size)
    pi.write(sensor_signal_pin, pigpio.LOW)


    for a in d:
        distance.append(int.from_bytes(a, byteorder='big', signed=False) / 2.0)
    distance = normalize(distance)
    print('distance:' + str(distance))

