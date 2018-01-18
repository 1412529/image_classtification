
import re
count=0;
product=[]
frequent=[]
support=[]
with open("frequent.txt") as f:
    for line in f:
        
        str= re.findall(r"[\w']+", line)
        if(len(str)>1):
            product.append(str[0])
            x=int(str[1])
            frequent.append(x)
        else:
            n=int(str[0])

for i in range(len(product)):
    frequent1=float(frequent[i])
    support.append(round(frequent1/(n*2),2));
print(product)
print(support)
       
for i in range(len(product)-1):
    for j in range(i,len(product)):
        if(product[i]>product[j]):
           temp=product[i]
           product[i]=product[j]
           product[j]=temp
           temp1=support[i]
           support[i]=support[j]
           support[j]=temp1

print(product)
print(support)
with open('generate_frequent.txt', 'a') as the_file:  
    for i in range(len(product)):
        the_file.write("MIS(")
        the_file.write("%s"%product[i])
        the_file.write(") = ")
        the_file.write("%s\n"%support[i])
#print(product)
#print(frequent)
#print(23)
#with open('frequent.txt', 'a') as the_file:
    
#print(str(frequent[0]));