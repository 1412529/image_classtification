
import re
count=0;
product=[]
frequent=[]
with open("data-1.txt") as f:
    for line in f:
        count=count+1;
        str= re.findall(r"[\w']+", line)
        for element in str:
            if(element in product):
                index=product.index(element);
                frequent[index]=frequent[index]+1
            else:
                product.append(element);
                frequent.append(1);

#print(product)
#print(frequent)
#print(23)
with open('frequent.txt', 'a') as the_file:
    for i in range(len(product)):
         the_file.write(product[i])
         the_file.write(": ")
         the_file.write("%s\n" % frequent[i])
     
    the_file.write("%s\n"%count);    
#print(str(frequent[0]));