import os
import json

sign_language_data = os.path.join("signLanguageData","تعارف.json")



with open(sign_language_data, 'r') as json_file:
    json_data = json.load(json_file)

    for key,value in json_data.items():
        print(key,value['word'])