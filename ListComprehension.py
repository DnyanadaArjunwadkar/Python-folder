# -*- coding: utf-8 -*-
"""
numbers = [32.2, -26.4, 32, 72.6, -18, 0.53, 12.7]
(a) Using a list comprehension, create a new list called temp out of the list numbers, which contains only
the positive numbers from the list, as integers. (b) Print the unique number in list temp in decreasing order.
"""
numbers = [32.2, -26.4, 32, 72.6, -18, 0.53, 12.7]

list_int=[]
for i in range(len(numbers)):
    list_int.append(int(numbers[i]))
for j in range(len(list_int)):
    list_int = [x for x in list_int if x >=0]  
    list_int.sort() 
    list_int.reverse()

#list comprehension part 1     
temp=[]
temp=[int(number) for number in numbers if int(number)>=0]
temp.sort()
temp.reverse()
print'______________________________________________________'
print temp
print'______________________________________________________'
print list_int

"""
Output>>>>>>>>>>>>>>>
______________________________________________________
[72, 32, 32, 12, 0]
______________________________________________________
[72, 32, 32, 12, 0]
"""    