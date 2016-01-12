

avg=float(raw_input('Enter your marks :'))
print avg

if avg>=93.33 and avg<100:
 print 'Congratulations! you have Aced it: grade A'
 
elif 90 <= avg < 93.33:
  print 'Congratulations!: grade A'

elif 86.667 <= avg < 90:
  print 'grade B+'

elif 83.33 <= avg < 86.667:
  print 'grade B'
  
elif 80 <= avg < 83.33:
  print 'grade B-'
  
elif 76.667 <= avg < 80:
  print 'grade C+'
  
elif 73.33 <= avg < 76.667:
  print 'grade C'
  
elif 70 <= avg < 73.33:
  print 'grade C-'
  
elif 66.667 <= avg < 70:
  print 'grade D+'
  
elif 65 <= avg < 66.667:
  print 'grade D'
  
elif avg < 65 and avg>0:
  print 'grade F'
 
else:
  print'Invalid input. please enter again'