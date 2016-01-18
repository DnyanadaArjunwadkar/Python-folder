# -*- coding: utf-8 -*-
"""
Write a function combine_lists_into_dic that combines these lists into a dictionary (note that
names[i]’s age is ages[i]). Then, write a function people that takes in an age and returns the names
of all the people who are that age.
"""
import sys
names = ['Alice', 'Bob', 'Cathy', 'Dan', 'Ed', 'Frank','Gary', 'Helen', 'Irene', 'Jack', 'Kelly', 'Larry']
ages = [18, 21, 18, 18, 19, 20, 20, 19, 19, 19, 22, 19]

def main():
    print'preparing dictionary..'
    combine_lists_into_dic(names,ages)
    operate()
    
def combine_lists_into_dic(lista,listb):
    global dictionary,val_list
    dictionary=dict(zip(names,ages))
    
def operate():
    search_age=int(raw_input('Enter the age'))
    print dictionary
    val_list=list(dictionary.values())
    #print val_list
    for key in dictionary.keys():
        
        if dictionary[key]==search_age:
            #print'Following people are %d years old' % search_age
            sys.stdout.write(key)
            sys.stdout.write(" ")
        elif search_age not in val_list:
            flag=1           
               
        else:
            dny='Happy'    
    if flag==1:
        print'age not found in the dictionary'
    else:
        dny='Happy'     
main()    
    

