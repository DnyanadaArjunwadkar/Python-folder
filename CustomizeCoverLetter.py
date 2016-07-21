
import datetime
import os

now = datetime.datetime.now()
now.strftime("%B %d, %Y")
str=now.strftime("%B %d, %Y")
from collections import Counter


def main():
    print 'Begun'

    company_name=raw_input('Company you are applying to :')
    position_name=raw_input('Position you are applying to :')
    
    f1=open('/Users/dny/Desktop/JOBs/BaseCoverLetter.txt','r')
    
    file_open1 = open('temp', 'w')
      
    for line in f1:
        file_open1.write(line.replace('ccc',company_name))
      
    f2=open('temp','r')
    file_open1 = open('temp2', 'w')

    for line in f2:
        file_open1.write(line.replace('ppp',position_name))

    destination_file='/Users/dny/Desktop/JOBs/'+company_name+'.txt'
    f3=open('temp2','r')
    file_open1 = open(destination_file, 'w')
    for line in f3:
        file_open1.write(line.replace('ddd',str))

    file_to_del=company_name+'.txt'
    str_command='rst2pdf'+" "+company_name+'.txt'+" "+company_name+".pdf"
    f1.close() 
    file_open1.close()
   
    os.system(str_command)
      
    os.remove(file_to_del)


    path = '/Users/dny/Desktop/JOBs/'
    img_list = os.listdir(path)
    for i in range(1,len(img_list)):
        if(img_list[i]=='temp'):
            os.remove(img_list[i])
        if(img_list[i]=='temp2'):
            os.remove(img_list[i])

    print 'done'

main()