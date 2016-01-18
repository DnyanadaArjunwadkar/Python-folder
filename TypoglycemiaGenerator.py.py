# -*- coding: utf-8 -*-
"""
Write a function scramble_words that implements a Typoglycemia generator. Do not
scramble the first and last letter in a word, but randomly scramble its interior letters.
Input: a line of words separated by blank space, e.g., 'Only smart people can read this'
Output: scrambled words, e.g., 'Olny sarmt ppoele can raed tihs'
"""
import string
import random
import sys
def display(text):
    dny='happy'
    #print text
     
def  scramble_words(text):
    global words,s,random_slice_joined
    global required_format
     
    #print'entered scramble_words'
    #print len(s)
    for x in s:
       # print('------------------------')
       # print x
        predicate = lambda m:m not in string.punctuation
        k=filter(predicate, x)
       # print 'without puctuation'
       # print k
        listx=list(k)
        limit=len(list(k))
        lim=limit-1
        slice=listx[1:lim]
       # print slice
        if len(slice)==1:
            required_format=listx
            random_slice_joined=''.join(listx)
            sys.stdout.write(random_slice_joined)
            sys.stdout.write(" ")
            
            #print '***********************'
           
        else:
            random.shuffle(slice,random.random)
            random_slice_joined=''.join(slice)
            required_format=listx[0]+ random_slice_joined + listx[lim]
            #print required_format
            sys.stdout.write(required_format)
            sys.stdout.write(" ")
            display(required_format)
          
def main():
    
    global words,s,random_slice_joined,cnt
    final_list=[]
    inputString= raw_input('Enter a test string :')
    s=inputString.split()
    print 'here'
    print s
    cnt=len(s)
    print cnt
    scramble_words(s)
main()