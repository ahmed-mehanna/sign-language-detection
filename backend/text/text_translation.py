from ast import Load
import os,json,re

SIGN_LANGUAGE_DATA_DIRECTORY = "signLanguageData"
SIGN_LANGUAGE_LETTERS_FILE = "signLanguageLetters.json"

class Loader:
    ''' this class is used to load signs and letters from json files to python dictionary
        any any changes in the logic or file structures should be handled only in this class
    '''
    replaced_letters_dic = {
        "ة":"ه", #  convert ه to ة
        "أ":"ا",
        "آ":"ا",
        "اّ":"ا",
        "إ":"ا",
        "ى":"ي"
    }


    def read_json(path):
        with open(path, 'r') as json_file:
            json_data = json.load(json_file)
            return json_data
        
        
    def process_word(word):
        for key,value in Loader.replaced_letters_dic.items():
            word = word.replace(key,value)
        
        return word
        
    
    def load_multi_file(path="",dic={}):
        ''' load json file where the value itself is a dictionary '''
        json_data = Loader.read_json(path)
        for sign,value in json_data.items():
            for word in value['words']:
                new_word = Loader.process_word(word)
                dic[new_word] = sign
        return dic
    
    
    def load_single_file(path="",dic={}):
        ''' load json file where the value is just a word '''
        json_data = Loader.read_json(path)
        for sign,letter in json_data.items():
            dic[letter] = sign
        return dic

    def load_signs(directory="",word_dic={}):
        for file_name in os.listdir(directory):
            path = os.path.join(directory,file_name)
            Loader.load_multi_file(path,word_dic)
        return word_dic
            

word_dic = Loader.load_signs(SIGN_LANGUAGE_DATA_DIRECTORY)
letters_dic = Loader.load_single_file(SIGN_LANGUAGE_LETTERS_FILE)




# problems 
# - fix the probelm of ة and ه

class Transformer:
    ''' this class contains multiple methods & transformation applied on the text '''
    replaced_letters_dic = {
        "ة":"ه", #  convert ه to ة
        "أ":"ا",
        "آ":"ا",
        "اّ":"ا",
        "إ":"ا",
        "ى":"ي"
    }
    
    def remove_duplications(text):
        p_longation = re.compile(r'(.)\1+')
        subst = r"\1\1"
        text = text.replace("\n"," ")
        text = re.sub(p_longation, subst, text)
        text = text.strip().replace("  "," ")
        
        text = text.replace('وو', 'و')
        text = text.replace('يي', 'ي')
        text = text.replace('اا', 'ا')
        
        return text
    
    def replace_letters(text):
        # convert list to string to apply the next operations
        
        for key,value in Transformer.replaced_letters_dic.items():
            text = text.replace(key,value)
        
        return text

            
    def pipeline(text,view=False):
        text = Transformer.remove_duplications(text)
        text = Transformer.replace_letters(text)
        final =  text.split(" ")
        
        if view:
            print("after text cleaning ",final)
        return final

        

class Synonyms:    
    ''' this class responsible of converting some words to their synonyms usually it work with 
        word that converted to multiple words
    '''
    
    synonyms_dic = {
        "ازيك" : ["كيف","حالك"],
        "و":"+",
    }
    
    def transform(lis,view=False):
        final = []
        for word in lis:
            if word in Synonyms.synonyms_dic:
                final.extend(Synonyms.synonyms_dic[word])
            else:
                final.append(word)
                
        if view:
            print("after synonyms",final)
        return final
    
    

class ArabicRoot:
    ''' ArabicRoot class is responsible to apply transformation on word to remove prefix or suffix 
        or split the word into multiple words ex : طعامك --> طعام + ملكك
    '''
    
    def __init__(self,word_list):
        self.word_list = word_list
        self.plural = ['ات',"ون","ين","ان"]
        self.mine = ['ني',"تي"]
        

        self.pre6 = ['كال',"بال","فال","مال","ولل","است","يست","تست","مست","وال"]
        self.pre5 = ["سن","ست","سى","لي","لن","لت","لل"]
        self.pre4 = ["ت","ي","ب","ل"]



        self.suf5 = ["ون","ات","ان","ين","تن","تم","كن","كم","هن","هم","يا","ني","تي","وا","ما","نا","ية","ها","اء"]
        self.suf4 = ['ت',"ة","ا","ي"]
        
        # list of extra methods (methods retur the word + other words)
        self.extra_methods = [self.extra_plural,self.extra_feminine,self.extra_ownership]
    
    def remove_prefix(self,word):
        ''' remove prefix from words ex : تشرب --> شرب '''
        if(len(word)>=6):
            for pre in self.pre6:
                if word.startswith(pre):
                    return word[3:]
        if(len(word)>=5):
            for pre in self.pre5:
                if word.startswith(pre):
                    return word[2:]
        if(len(word)>=4):
            for pre in self.pre4:
                if word.startswith(pre):
                    return word[1:]
        
        return word
                
    
    def remove_suffix(self,word):        
        ''' remove suffix from words ex : ياكلون --> ياكل  '''
        if(len(word)>=5):                
            for suf in self.suf5:
                if word.endswith(suf):
                    return word[:-2]
        if(len(word)>=4):
            for suf in self.suf4:
                if word.endswith(suf):
                    return word[:-1]
        
        return word
    
    def clean_word(self,word):
        ''' remove both suffix and prefix '''
        return self.remove_suffix(self.remove_prefix(word))
        
    
    
    
    def extra_plural(self,word):
        ''' if the word is plural then convert it to word +  كثير but i need 
            to make some changes in it to handle verbs being plural when word are plural 
            ex : ['مهندس', 'كثير', 'يكره', 'كثير', 'جامعة']
            also need to return word + كثير even if word not in word_list
        '''
        if(len(word)>=5):                
            for suf in self.plural:
                if word.endswith(suf) and word[:-2] in self.word_list:
                    return [word[:-2],"كثير"]
        return []
    
    def extra_feminine(self,word):
        if(word[-1]=='ة' and word[:-1] in self.word_list):
            return [word[:-1],"مؤنث"]
        return []
    
    def extra_ownership(self,word):
        if(word[-1]=='ي' and word[:-1] in self.word_list):
            return [word[:-1],"ملكى"]
        if(word[-1]=='ك' and word[:-1] in self.word_list):
            return [word[:-1],"ملكك"]
        return []
    
    def get_extra_words(self,word):
        ''' call all extra_ functions to get the word + extra meaning '''
        
        for method in self.extra_methods:
            output = method(word)
            if(len(output)>0):return output

        
        return []
    
# fill it later

class GroupWords:
    
    def transform(lis,view=False):
        # TODO fill this function later
        final = []
        i=0
        while i < len(lis) - 1:
            grouped_word = lis[i]+" "+lis[i+1]
            if grouped_word in word_dic:
                final.append(grouped_word)
                i+=1
            else:
                final.append(lis[i])
            i+=1
        if i < len(lis):
            final.append(lis[-1])
        print("after Grouping ",final)
        return final



class Translation:
    ''' this class is responsible of translation applying final transformation on words to select the 
        correct words from the dictionary
    '''
    def __init__(self,word_dic):
        self.word_dic = word_dic
        self.arabic_root = ArabicRoot(word_dic)
        
    def translate(self,lis,view=False):
        final = []
        for word in lis:
            if word in self.word_dic:
                final.append(word)
                continue

            if(len(word)>3 and word[:2]=="ال"):
                word = word[2:]
                if word in self.word_dic:
                    final.append(word)
                    continue


            if(len(word)>=6 and word[:3]=="وال"):
                word = word[3:]
                final.append('+')
                if word in self.word_dic:
                    final.append(word)
                    continue

            output = self.arabic_root.get_extra_words(word)
            if(len(output)>0):
                final.extend(output)
                continue

            final_form = self.arabic_root.clean_word(word)
            if(final_form in self.word_dic):
                final.append(final_form)
                continue



            final.append(word)


        if(view):
            print("after translation ",final)
        return final



class Signs:
    ''' this class is responsible of getting the signs '''
    def __init__(self,word_dic):
        self.word_dic = word_dic
    
    
    def get_letters(self,lis,view=False):
        final = []
        for word in lis:
            
            if word in self.word_dic:
                final.append(word)
                continue
                
            if(len(final)>1 and (len(final[-1])==1 and final[-1]!='+' and word!='+') ):
                 final.append(" ")
            for letter in word:
                final.append(letter)
                

        if(view):
            print("after letters ",final)
        return final


    def get_signs(self,lis,view=False):
        final =[]
        for word in lis:
            if len(word)>1:
                final.append(self.word_dic[word])
            else:
                final.append(letters_dic[word])
                
        if(view):
            for sign in final:
                print(sign)
        return final
    
    
    def pipeline(self,lis,view=False):
        lis = self.get_letters(lis,view)
        lis = self.get_signs(lis,view)
        return lis
    

translation = Translation(word_dic)
sign_obj = Signs(word_dic)


def pipeline(s):
    output = Transformer.pipeline(s,view=True)
    
    output = GroupWords.transform(output,view=True)
    
    output = translation.translate(output,view=True)
    
    output = Synonyms.transform(output,view=True)
    
    output = GroupWords.transform(output,view=True)
    
    output = sign_obj.pipeline(output,view=True)
    
    
    
    return output

