import json 

# dataset partition
indrange_train = []
indrange_test = []
indrange_validation = []

for x in range(180):
  if x % 10 < 8 :
    indrange_train.append(x)

  if x % 10 == 9:
    indrange_test.append(x)

  if x % 20 == 18:
    indrange_validation.append(x)

  if x % 20 == 8:
    indrange_test.append(x)

with open('./dataset/data_split.json','w') as jf:
    json.dump({'train':indrange_train,'valid':indrange_validation,'test':indrange_test},jf)

