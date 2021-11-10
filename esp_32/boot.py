# This file is executed on every boot (including wake-boot from deepsleep)
import esp
#esp.osdebug(None)
import network

ap = network.WLAN(network.AP_IF) # create access-point interface
ap.config(essid='ESP-AP') # set the ESSID of the access point
ap.config(max_clients=10) # set how many clients can connect to the network
ap.active(True)


