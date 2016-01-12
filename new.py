
def main():
    lista={}
    x=3
    y=2
    lista[0]=x
    lista[1]=y
    print lista
    file_open1 = open('file1.txt', 'w')
    for item in lista:
     file_open1.write("%s\t" % item)
    
    print'dine'
main()
     