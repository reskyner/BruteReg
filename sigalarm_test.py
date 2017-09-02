import signal

class TimeoutException(Exception):   # Custom exception class
  pass

def TimeoutHandler(signum, frame):   # Custom signal handler
  raise TimeoutException

# Change the behavior of SIGALRM
OriginalHandler = signal.signal(signal.SIGALRM,TimeoutHandler)

# Start the timer. Once 30 seconds are over, a SIGALRM signal is sent.
signal.alarm(1)

# This try/except loop ensures that you'll catch TimeoutException when it's sent.

counter = 1
try:
  while counter==1:
      print('hello')
except TimeoutException:
  print "SSH command timed out."

# Reset the alarm stuff.
signal.alarm(0)
signal.signal(signal.SIGALRM,OriginalHandler)