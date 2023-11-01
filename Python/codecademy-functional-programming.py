import csv
from functools import reduce

def count(predicate, itr):
  # CP1
  count_filter = filter(predicate, itr)
  # CP2
  count_reduce = reduce(lambda x, y: x + 1,count_filter, 0)
  return count_reduce

def average(itr):
  #CP 8
  iterable = iter(itr)
  return avg_helper(0, iterable, 0)
  
# CP 3/4/5/6
def avg_helper(curr_count, itr, curr_sum):
  next_num = next(itr, "null")
  if next_num == "null":
    return curr_sum/curr_count
  curr_sum += next_num
  curr_count += 1
  return avg_helper(curr_count, itr, curr_sum)

with open('1kSalesRec.csv', newline = '') as csvfile:
  reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  fields = next(reader)
  count_belgiums = count(lambda x: x[1] == "Belgium", reader)
  print(count_belgiums)
  csvfile.seek(0)
  avg_portugal = average(map(lambda x: float(x[13]), filter(lambda x: x[1] == "Portugal", reader)))
  print(avg_portugal)