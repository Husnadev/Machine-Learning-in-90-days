'''abc = 'Hussain'
print(abc.isalnum())
print(abc.isalpha())
print(abc.istitle())
print(abc.isdigit())
print(abc.isupper())
print(abc.islower())
print(abc.isspace())
print(abc.startswith('h'))
print(abc.endswith('n'))

a = "Hussain"
b = "Tokare"
print(a.isalpha() or b.isnumeric())
print(a.isalpha() and b.isnumeric())

#print(type([]))
my_list = ['hussain', 'tokare', 100, 200, 300]
#print(type(my_list))
#print(len(my_list))


#Append :
my_list.append('hussyboltz')
#print(my_list[3])
#print(my_list[1:3])

#NEsted List
my_list.append(['abc','def'])
#print(my_list)
#print(len(my_list))

#Insert :
my_list.insert(2,'hussy')
my_list.insert(7,'omn')
#print(my_list)
#print(len(my_list))


#Extend :
my_list.extend(['erg','erf'])
#print(len(my_list))

#Operations :
lst = [1,1,2,3,4,5]
#print(lst)

#Sum :
#print(sum(lst))

#POP :
#print(lst.pop(2))
print(lst.pop(3))
print(lst)

#count: 
print(lst.count(1))

print(lst.pop(1))
print(lst)

#MAx and Min :
print(min(lst))
print(max(lst))


#SETS
#set_var  = set()
#print(type(set_var))

#set_var = {'a','b','c','d',2,3,4,5,'a'}
#print(len(set_var))

#sets = {'hussain','rohit','kaif'}
#print(sets)
#ADD :

#sets.add('husna')
#print(sets)

#Difference 
#set1 = {'a','b','c','d'}
#set2 = {'a','b','c','d','e'}
#print(set2.difference(set1))
#print(set2)

#Difference_update
#set1 = {'a','b','c','d'}
#set3 = {'a','b','c','d','e'}
#print(set3.difference_update(set1))

#Intersection :
set4 = {'a','b'}
set5 = {'a','b','c'}
#print(set5.intersection(set4))
#print(set5)

#Intersection_update :
#print(set5.intersection_update(set4))
#print(set5)

'''
'''
#Dictionaries :
dic = {}
print(type(dic))

dict = {'k1':'hussain',
        'k2':'kaif',
        'k3':'rohit'}
print(dict)
print(dict['k2'])

dict['k4'] = 'yawn'
dict['k1']='hussy'
print(dict)

#Nested Dictionaries

car_type1 = {'a1':'abc','a2':'def'}
car_type2 = {'a1':'efg','a2':'hij'}
car_type3 = {'a1':'uvw','a2':'sto'}

car_type = {
    'model1' : car_type1,
    'model2' : car_type2,
    'model3' : car_type3
}

print(car_type['model3']['a1'])

for x in car_type:
    print(x)

for y in car_type.values():
    print(y)    

for z in car_type.items():
    print(z)    
'''    
'''
#Tuple :
#tup = tuple()
#print(type(tup))

tup = ('a','b','c',1,2,3,4,2)
print(tup[1])

#tup[1] = 'd'
print(tup)
print(tup.index(1,0,4))
print(tup.count(2))
'''

