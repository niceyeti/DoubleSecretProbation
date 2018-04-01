from bluetooth import *
from pprint import pprint


print "begin..."
devices = discover_devices()
print "devs: ",devices
service = find_service("48:59:29:0A:41:D3")
print "services:"
pprint(service)

