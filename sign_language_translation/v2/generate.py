import os
import json

f1 = os.path.join("signLanguageWords","تعارف.txt")
f2 = os.path.join("signLanguageWords","منزل_عائلة.txt")
f3 = os.path.join("signLanguageWords","ثالث.txt")



sign_language_data = os.path.join("signLanguageData","تعارف.json")


json_data = {}
existing_words = {}
counter = 0;

with open(sign_language_data, 'r') as json_file:
    json_data = json.load(json_file)
    counter = len(json_data)
    for key,value in json_data.items():
        existing_words[value['word']] = 1;


    
with open(f1,"r") as d1 ,open(f2,"r") as d2,open(f3,"r") as d3:
    lis = d1.read().split("\n")
    lis.extend(d2.read().split("\n"))
    lis.extend(d3.read().split("\n"))
    
    
    
    for word in lis:
        if(word not in existing_words):
            json_data[f'sign_{counter}'] = {"word":word,"description":"","synonyms":[]}
            counter+=1
    with open(sign_language_data, 'w') as json_file:
        json.dump(json_data,json_file)




for key,value in json_data.items():
    print(key,value['word'])