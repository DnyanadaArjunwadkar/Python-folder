p1=0
p2=0
count=0    
def main():
    playernames()
    print'player names done'
    shape_p1()
    print'*ended*'
  
def playernames():
      global player1
      global player2 
      player1=raw_input('Enter your name player 1 :')
      player2=raw_input('Enter your name player 2 :')
      
def shape_p1():
        global s1
        
        global count
        count +=1
    
        if count>3:
          print'count crossed 3'  
        else:
                
          s1=raw_input('Enter your Shape player 1 :')
          
          while s1 in ('rock','paper','scissors'):
              shape_p2()
              break
              if s1 not in ('rock','paper','scissors'):
                 break           
      
def shape_p2():
    global s2
  
    s2=raw_input('Enter your Shape player 2 :')
    while s2 in ('rock','paper','scissors'):
              check()
              break
              if s2 not in ('rock','paper','scissors'):
                 break           

def check():
    global p1
    global p2
    global count
    if(s1==s2):
     print'draw'
     count=count-1
     shape_p1()
    elif(s1=='rock' and s2=='scissors'):
     p1=p1+1
     shape_p1()
    
    elif(s1=='rock' and s2=='paper'):
     p2=p2+1
     shape_p1()
    
    elif(s1=='paper' and s2=='rock'):
     p1=p1+1
     shape_p1()
    
    elif(s1=='paper' and s2=='scissors'):
     p2=p2+1
     shape_p1()
    
    elif(s1=='scissors' and s2=='paper'):
     p1=p1+1
     shape_p1()
    
    elif(s1=='scissors' and s2=='rock'):
     p2=p2+1
     shape_p1()
    
    else: 
     print'go home '
    
    result()

def result():
    global p1
    global p2
    global player1
    global player2
    
    if(p1>p2):
      print'%s wins'% player1
    else:
      print'%s wins'% player2

      
main()    