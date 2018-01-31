
import re
n=0
count=0;
count1=0
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
    if(frequent[i]>6000):
        support.append(frequent1/(n*20));
        count1=count1+1;
    else:
        support.append(1.0);
#print(product)
#print(support)
print(count1)   
for i in range(len(product)-1):
    for j in range(i,len(product)):
        if(product[i]>product[j]):
           temp=product[i]
           product[i]=product[j]
           product[j]=temp
           temp1=support[i]
           support[i]=support[j]
           support[j]=temp1

#print(product)#
#print(support)
with open('generate_frequent.txt', 'a') as the_file:  
    the_file.write("SDC = ")
    sdc=float(200)/(n)
    the_file.write("%s"%sdc)
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