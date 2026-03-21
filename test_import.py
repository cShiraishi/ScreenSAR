import sys
import traceback
import faulthandler
import threading

def dump_trace():
    print("--- DUMPING TRACE ---")
    faulthandler.dump_traceback()
    sys.exit(1)

timer = threading.Timer(10.0, dump_trace)
timer.start()

print("Importing auth...")
import src.ui.auth
print("Done importing auth.")
timer.cancel()
