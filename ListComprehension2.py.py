"""
Given a=[[1,2],[3,4,5],[6]],
Create a new list [1,2,3,4,5,6] usingnlist comprehension. 
"""
a=[[1,2],[3,4,5],[6]]
newlist=[num for element in a for num in element]
print newlist
"""
Output>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
_____________________________________________
[1, 2, 3, 4, 5, 6]

"""