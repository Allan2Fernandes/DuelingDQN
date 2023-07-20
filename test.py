from collections import deque

# Example deque
my_deque = deque(maxlen=5)
my_deque.append(0)
my_deque.append(1)
my_deque.append(2)
my_deque.append(3)
my_deque.append(4)
my_deque.append(10)
# Convert deque to a list and get all elements from oldest to the second newest
elements_from_oldest_to_second_newest = list(my_deque)[:-1]

print(elements_from_oldest_to_second_newest)