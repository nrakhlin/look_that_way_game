import pigpio
from time import sleep

# Need to run "sudo pigpiod" to start daemon before running
# Need to run "sudo killall pigpiod" to stop daemon when done

# Only GPIOs 12, 13, 18, and 19 have hardware PWM capabilities 
# --> issue because PiTFT uses 18, so if we want 2 camera gimbals
# we need 4 hardware PWMs.

# Use GPIOs 12 and 19 for finger gimbal
BASE_GPIO = 12
HEAD_GPIO = 19

# Set up hardware PWM values
duty_cycle = 500000     # 50% duty cycle
left_freq = 500         # 500 Hz @ 50% duty cycle = 1 ms pulse width
middle_freq = 334       # 334 Hz @ 50% duty cycle = 1.5 ms pulse width
right_freq = 250        # 250 Hz @ 50% duty cycle = 2 ms pulse width

pi = pigpio.pi()

# Try using hardware_PWM
pi.hardware_PWM(BASE_GPIO, left_freq, duty_cycle)
print("Left")
sleep(5)
pi.hardware_PWM(BASE_GPIO, middle_freq, duty_cycle)
print("Middle")
sleep(5)
pi.hardware_PWM(BASE_GPIO, right_freq, duty_cycle)
print("Right")
sleep(5)
pi.hardware_PWM(BASE_GPIO, 1, 0)
print("Off")

# Try using set_servo_pulswidth
# pi.set_servo_pulsewidth(BASE_GPIO, 1000) # safe anti-clockwise
# print("CCW")
# sleep(5)
# pi.set_servo_pulsewidth(BASE_GPIO, 1500) # centre
# print("center")
# sleep(5)
# pi.set_servo_pulsewidth(BASE_GPIO, 2000) # safe clockwise
# print("CW")
# sleep(5)
# pi.set_servo_pulsewidth(BASE_GPIO, 0)    # off
# print("Off")