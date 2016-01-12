"""
Write a Python program that reads a text file and reports statistics on the file. The statistics are as follows:
• The number of occurrences of the first 5 letters in in the alphabet (regardless of case). The frequency should also be reported as a percentage of each letter over all the non-whitespace characters in the text file (including numbers and punctuation). Note that the reporting for each letter is not case sensitive. For example, “A” and “a” should be counted as a single letter.
• The number of words in the text file.
• The number of non-whitespace characters in the text file.
Your program must utilize at least one user-defined function, which passes the required data into the function as arguments and returns data back from the function as return values. The results of your program should be written to a file, with the data for each word and each letter on its own line. The program should prompt the user for the input and output filenames.
Note: sample test data and program output can be found in separate files.
"""





import collections
import string

def main():
    global num_lines
    global num_words
    global num_chars
    print 'Please enter the full name of the desired file (with extension) at the prompt below'
    file = raw_input('What to read?: ')
    readstat(file)      
    file1 = raw_input('where to write result?: ')
    file_open1 = open(file1, 'w')
    writestat(file_open1) 
    print'read write done.closing files now..'
    file_open1.close()
    print'files closed'
    
def readstat(file):
    global num_words
    global num_chars 
    
    
    num_words=0
    num_chars=0
    
    file_open=open(file)
    f2=file_open.read()
    count_letters(f2)
        
    c = collections.Counter()
    with open(file, 'r') as f:
     for line in f:
        c.update(line.rstrip().lower())
        words = line.split()
        num_words += len(words)
        num_chars += len(line)
    lines = collections.Counter(file_open) 
    global k
    k= len(lines)
    print 'words,char and  '
    print num_words
    print num_chars
    print 'Most common:'
   
    for letter, count in c.items():
        print '%s: %7d' % (letter, count)

    
def count_letters(word):
    
     global aorA,borB,corC,dorD,eorE
     global ap,bp,cp,dp,ep,total
        
     count = collections.Counter(word) # this counts all the letters, including invalid ones
     ascii_letter=sum(count[letter] for letter in string.ascii_letters)
     digits=sum(count[letter] for letter in string.digits)
     punc=sum(count[letter] for letter in string.punctuation)
     total=punc+digits+ascii_letter
     arrA=["a","A"]
     aorA=sum(count[letter] for letter in arrA)
     ap=float(aorA*100)/total
     arrB=["b","B"]
     borB=sum(count[letter] for letter in arrB)
     bp=float(borB*100)/total
     
     arrC=["c","C"]
     corC=sum(count[letter] for letter in arrC)
     cp=float(corC*100)/total
     
     arrD=["d","D"]
     dorD=sum(count[letter] for letter in arrD)
     dp=float(dorD*100)/total
     
     arrE=["e","E"]
     eorE=sum(count[letter] for letter in arrE)
     ep=float(eorE*100)/total
     
     
     
     print'letter,digi,punc'
     print ascii_letter
     print digits
     print punc
     print 'Total++++++++++++++++'
     print total
     print 'a or A'   
     print aorA
     print ap
     print 'b or B'   
     print borB
     print bp
     print 'c or C'   
     print corC
     print cp
     print 'd or D'   
     print dorD
     print dp
     print 'e or E'   
     print eorE
     print ep    
     
def writestat(file_open1):
    global k
    global num_words
    global num_chars
    global aorA,borB,corC,dorD,eorE
    global ap,bp,cp,dp,ep,total
    
    file_open1.write('\nnumber of total words :%d'% num_words)
    #file_open1.write('\nnumber of char including whitespaces and newlines and special symbols:%d'% num_chars)
    file_open1.write('\ntotal non-white char: %d'% total)
    file_open1.write('\n_____________________________________________________________________________')
    
    file_open1.write('\nLetter\tFrequency\tPercentage%:\n')
    file_open1.write('\nA')
    file_open1.write('\t\t\t%d'% aorA)
    file_open1.write('\t\t\t%.2f\n'% ap)
    
    file_open1.write('\nB')
    file_open1.write('\t\t\t%d'% borB)
    file_open1.write('\t\t\t%.2f\n'% bp)
    
    file_open1.write('\nC')
    file_open1.write('\t\t\t%d'% corC)
    file_open1.write('\t\t\t%.2f\n'% cp)
    
    file_open1.write('\nD')
    file_open1.write('\t\t\t%d'% dorD)
    file_open1.write('\t\t\t%.2f\n'% dp)
    
    file_open1.write('\nE')
    file_open1.write('\t\t\t%d'% eorE)
    file_open1.write('\t\t\t%.2f\n'% ep)
    file_open1.write('\n_____________________________________________________________________________')
         
main()
