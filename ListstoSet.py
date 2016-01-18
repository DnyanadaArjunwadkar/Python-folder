# -*- coding: utf-8 -*-
"""
following two lists
names = ['Alice', 'Bob', 'Cathy', 'Dan', 'Ed', 'Frank',
 'Gary', 'Helen', 'Irene', 'Jack', 'Kelly', 'Larry']
ages = [18, 21, 18, 18, 19, 20, 20, 19, 19, 19, 22, 19]
Write a function combine_lists_into_dic that combines these lists into a dictionary (note that
names[i]’s age is ages[i]). Then, write a function people that takes in an age and returns the names
of all the people who are that age. Hint: make sure to test “edge conditions”, e.g., test on people(27)should
return an empty list.
Given the same lists in question 3., write a function combine_lists_into_set that
combines these lists into a set. Then, write a function add that takes two inputs:name and age, and return a
Boolean indicating if a person’s information is successfully added to the set. Hint: make sure to handle
exceptions if a person with the same name and age is already in the set.
"""
import sys
names = ['Alice', 'Bob', 'Cathy', 'Dan', 'Ed', 'Frank','Gary', 'Helen', 'Irene', 'Jack', 'Kelly', 'Larry']
ages = [18, 21, 18, 18, 19, 20, 20, 19, 19, 19, 22, 19]

def main():
    print'preparing set..'
    combine_lists_into_set(names,ages)
    add_fn()
    
def combine_lists_into_set(lista,listb):
    global list_set,my_dict
    list_zip=zip(names,ages)
    list_set=set(list_zip)
    my_dict=dict(list_zip)
    print list_set
    
def add_fn():
    name_ip=raw_input('Enter name to add:')
    age_ip=int(raw_input('Enter age to add:'))
    #combine_lists_into_set(names,ages)
   
    if name_ip not in names:
        names.append(name_ip)
        ages.append(age_ip)
        combine_lists_into_set(names,ages)
   
#        print names
    else:
       # for number in range(len(names)):
            s=my_dict[name_ip]
            if s==age_ip :
                print 'Duplicate detected'
            else:
                names.append(name_ip)
                ages.append(age_ip)
                combine_lists_into_set(names,ages)
               
   
main()    
    

