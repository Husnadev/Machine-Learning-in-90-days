'''abc = 'Hussain'
#print(abc.isalnum())
#print(abc.isalpha())
#print(abc.istitle())
#print(abc.isdigit())
#print(abc.isupper())
#print(abc.islower())
#print(abc.isspace())
#print(abc.startswith('h'))
#print(abc.endswith('n'))

a = "Hussain"
b = "Tokare"
#print(a.isalpha() or b.isnumeric())
#print(a.isalpha() and b.isnumeric())

##print(type([]))
my_list = ['hussain', 'tokare', 100, 200, 300]
##print(type(my_list))
##print(len(my_list))


#Append :
my_list.append('hussyboltz')
##print(my_list[3])
##print(my_list[1:3])

#NEsted List
my_list.append(['abc','def'])
##print(my_list)
##print(len(my_list))

#Insert :
my_list.insert(2,'hussy')
my_list.insert(7,'omn')
##print(my_list)
##print(len(my_list))


#Extend :
my_list.extend(['erg','erf'])
##print(len(my_list))

#Operations :
lst = [1,1,2,3,4,5]
##print(lst)

#Sum :
##print(sum(lst))

#POP :
##print(lst.pop(2))
#print(lst.pop(3))
#print(lst)

#count: 
#print(lst.count(1))

#print(lst.pop(1))
#print(lst)

#MAx and Min :
#print(min(lst))
#print(max(lst))


#SETS
#set_var  = set()
##print(type(set_var))

#set_var = {'a','b','c','d',2,3,4,5,'a'}
##print(len(set_var))

#sets = {'hussain','rohit','kaif'}
##print(sets)
#ADD :

#sets.add('husna')
##print(sets)

#Difference 
#set1 = {'a','b','c','d'}
#set2 = {'a','b','c','d','e'}
##print(set2.difference(set1))
##print(set2)

#Difference_update
#set1 = {'a','b','c','d'}
#set3 = {'a','b','c','d','e'}
##print(set3.difference_update(set1))

#Intersection :
set4 = {'a','b'}
set5 = {'a','b','c'}
##print(set5.intersection(set4))
##print(set5)

#Intersection_update :
##print(set5.intersection_update(set4))
##print(set5)

'''
'''
#Dictionaries :
dic = {}
#print(type(dic))

dict = {'k1':'hussain',
        'k2':'kaif',
        'k3':'rohit'}
#print(dict)
#print(dict['k2'])

dict['k4'] = 'yawn'
dict['k1']='hussy'
#print(dict)

#Nested Dictionaries

car_type1 = {'a1':'abc','a2':'def'}
car_type2 = {'a1':'efg','a2':'hij'}
car_type3 = {'a1':'uvw','a2':'sto'}

car_type = {
    'model1' : car_type1,
    'model2' : car_type2,
    'model3' : car_type3
}

#print(car_type['model3']['a1'])

for x in car_type:
    #print(x)

for y in car_type.values():
    #print(y)    

for z in car_type.items():
    #print(z)    
'''    
'''
#Tuple :
#tup = tuple()
##print(type(tup))

tup = ('a','b','c',1,2,3,4,2)
#print(tup[1])

#tup[1] = 'd'
#print(tup)
#print(tup.index(1,0,4))
#print(tup.count(2))
'''
import numpy as np
listy = [1,2,3,4,5]
##print(type(listy))
arr = np.array(listy)
##print(arr)
##print(len(arr))
##print(arr[3])
##print(type(arr))

##print(arr.shape)

list1 = [1,2,3,4,5]
list2 = [5,4,6,7,8]
list3 = [2,3,4,5,9]

arr = np.array([list1,list2,list3])
##print(arr.shape)
##print(arr.reshape(3,5))

##print(arr[1:,2:4])
##print(arr[1:2,1:4])

arr = np.array([1,2,3,4,5,6,7,8])
##print(type(arr))
arr[3:]=100
##print(arr)
arr1 = arr
arr1[3:]=500
##print(arr1)
##print(arr)
arr1 = arr.copy()
##print(arr1)
arr1[3:] = 1000
##print(arr1)
##print(arr)


arr = np.array([1,2,3,4,5])
val = 2
##print(arr[arr<val])
##print(arr*val)


arr1 = np.arange(0,10,step=1,dtype=int).reshape(2,5)
arr2 = np.arange(0,10,step=1,dtype=int).reshape(2,5)
##print(arr1*arr2)


#arr = np.random.rand(3,5).reshape(5,3) - > Normal Distribution
#arr = np.random.randn(4,4).reshape(8,2) - > Standard Normal Distribution
#arr = np.random.randint(1,100)  #.reshape(4,2) - > Discrete Uniform Distribution
#arr = np.ones([3,4],dtype=int)
arr = np.random.random_sample([3,4])
##print(arr)
#np.random.randint()

#Pandas

import pandas as pd
df = pd.DataFrame(np.arange(0,20).reshape(5,4),index=['r1','r2','r3','r4','r5'],columns=['c1','c2','c3','c4'],dtype=int)
##print(df.head())
##print(df['c1'])
##print(df[['c1','c2']])

#LOC
##print(df.loc['r2'])

#ILOC:
##print(df.iloc[1:3,1:3])
##print(df.iloc[0,:])

#converting datafram to list :
##print(type(df.iloc[:,:].values))

##print(df.isnull().sum())

##print(df.loc[0,:].value_counts())

#df.to_csv('test1.csv')


#Pandas Part Two

df = pd.read_csv('mercedesbenz.csv')
##print(df.head())
##print(df.info())
##print(df.describe())

from io import StringIO, BytesIO
data = pd.read_csv('test1.csv',sep=';',index_col=0)
##print(data)
#print(df.head())
#print(df['X0'].value_counts())
#print(df[df['y']>100])

data = (
    'c1,c2,c3\n'
    'r1,x,y,1\n'
    'r2,a,b,2\n'
    'r3,c,d,3'
)
df = pd.read_csv(StringIO(data),dtype={'c1':object,'c2':object,'c3':float})
#print(df)
#print(df.dtypes)

data = (
    'index,c1,c2,c3\n'
    '4,1,2,3\n'
    '8,7,6,9'
)
df = pd.read_csv(StringIO(data),index_col=False)
#print(df)

data = (
    'c1,c2,c3\n'
    '1,2,3\n'
    '7,6,9'
)
df = pd.read_csv(StringIO(data),index_col=False,usecols=['c1','c3'])
#print(df)

data = (
    'a,b,c\n'
    '0,"Hello",\\"Bob-builder"\\,"Good"'
)

#print(pd.read_csv(StringIO(data),escapechar='\\'))

data = '{"name":"hussain","sex":"Male","status":"Married","Hobby":[{"a":1,"b":2}]}'
df = pd.read_json(data,orient="index")


df1 = pd.read_json('EmployeeData.json')
df1.to_csv('wow.csv')
df3 = pd.read_csv('wow.csv')
df4 = df3.to_json(orient='index')
#print(df4)
import html5lib
df = pd.read_html('https://www.countrycode.org/',header=0)
#print(df)

df100 = pd.read_excel('Book2.xlsx')

print(type(df100))