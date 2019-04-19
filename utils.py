import sys
import time
def print_time(start_time, f=None):
  """Take a start time, print elapsed duration, and return a new time."""
  s = "time %ds, %s." % ((time.time() - start_time), time.ctime())
  print(s)
  if f:
    f.write(s.encode("utf-8"))
    f.write(b"\n")
  sys.stdout.flush()
  return time.time()


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  # print(s.encode("utf-8"), end="", file=sys.stdout)
  print(s.encode("utf-8"))
  # if new_line:
  #   sys.stdout.write("\n")
  sys.stdout.flush()



def print_hparams(hparams, skip_patterns=None, f=None):
  """Print hparams, can skip keys based on pattern."""
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])), f)


def normal_leven(str1, str2):
  len_str1 = len(str1) + 1
  len_str2 = len(str2) + 1
  # create matrix
  matrix = [0 for n in range(len_str1 * len_str2)]
  # init x axis
  for i in range(len_str1):
    matrix[i] = i
  # init y axis
  for j in range(0, len(matrix), len_str1):
    if j % len_str1 == 0:
      matrix[j] = j // len_str1

  for i in range(1, len_str1):
    for j in range(1, len_str2):
      if str1[i - 1] == str2[j - 1]:
        cost = 0
      else:
        cost = 1
      matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                     matrix[j * len_str1 + (i - 1)] + 1,
                                     matrix[(j - 1) * len_str1 + (i - 1)] + cost)

  return matrix[-1]
