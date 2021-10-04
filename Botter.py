import os
import pandas as pd
import nltk 
import numpy as np
import re
import time
import datetime
from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words
import spacy
import textacy.extract
import discord

DISCORD_BOT_TOKEN = ''

client = discord.Client()

@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')


# Загрузка английской NLP-модели
nlp = spacy.load('en_core_web_lg')



df=pd.read_excel('dialog_talk_agent.xlsx')
df.ffill(axis = 0,inplace=True)

def step1(x):
    for i in x:
        a=str(i).lower()
        p=re.sub(r'[^a-z0-9]',' ',a)
        print(p)
        
lemma = wordnet.WordNetLemmatizer()

def text_normalization(text):
    text=str(text).lower() # text to lower case
    spl_char_text=re.sub(r'[^ a-z]','',text) # removing special characters
    tokens=nltk.word_tokenize(spl_char_text) # word tokenizing
    lema=wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list=pos_tag(tokens,tagset=None) # parts of speech
    lema_words=[]   # empty list 
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val='v'
        elif pos_token.startswith('J'): # Adjective
            pos_val='a'
        elif pos_token.startswith('R'): # Adverb
            pos_val='r'
        else:
            pos_val='n' # Noun
        lema_token=lema.lemmatize(token,pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence 

df['lemmatized_text']=df['Context'].apply(text_normalization)

stop = stopwords.words('english')

cv = CountVectorizer() # intializing the count vectorizer
X = cv.fit_transform(df['lemmatized_text']).toarray()

tfidf=TfidfVectorizer() # intializing tf-id 
x_tfidf=tfidf.fit_transform(df['lemmatized_text']).toarray()
 
Question ='Tell me about yourself.'
Question_lemma = text_normalization(Question)
Question_tfidf = tfidf.transform([Question_lemma]).toarray() # applying tf-idf

# using tf-idf

df_simi_tfidf = pd.DataFrame(df, columns=['Text Response','similarity_tfidf']) 
df_tfidf=pd.DataFrame(x_tfidf,columns=tfidf.get_feature_names()) 
cos=1-pairwise_distances(df_tfidf,Question_tfidf,metric='cosine')

df['similarity_tfidf']=cos # creating a new column 

df_simi_tfidf_sort = df_simi_tfidf.sort_values(by='similarity_tfidf', ascending=False) 

threshold = 0.2 # considering the value of p=smiliarity to be greater than 0.2
df_threshold = df_simi_tfidf_sort[df_simi_tfidf_sort['similarity_tfidf'] > threshold] 

def chat_tfidf(text):
    lemma=text_normalization(text) # calling the function to perform text normalization
    tf=tfidf.transform([lemma]).toarray() # applying tf-idf
    cos=1-pairwise_distances(df_tfidf,tf,metric='cosine') # applying cosine similarity
    index_value=cos.argmax() # getting index value 
    return df['Text Response'].loc[index_value]

#def NLP_Tag(text):
# в переменной 'doc' теперь содержится обработанная версия текста
# мы можем делать с ней все что угодно!
# например, распечатать все обнаруженные именованные сущности
   

def fact(text, word):
    doc = nlp(text)
# Извлечение полуструктурированных выражений со словом Москва
    statements = textacy.extract.semistructured_statements(doc, word)

# Вывод результатов
    print("Here are the things I know about " + word + " :")

    for statement in statements:
        subject, verb, fact = statement
        print(f" - {fact}")

    
# Если токен является именем, заменяем его словом "REDACTED"  
def replace_name_with_placeholder(token):
    if token.ent_iob != 0 and token.ent_type_ == tik:
        return rep
    else:
        return token.string

# Проверка всех сущностей
def scrub(text): 
    doc = nlp(text)
    for ent in doc.ents:
        ent.merge
    tokens = map(replace_name_with_placeholder,doc)
    return "".join(tokens)


cmds = {
        'time',
        'jokes',
        'NLP_Tag',
        'NLP_Fact',
        'NLP_Placehold'
    }

@client.event
async def on_message(message):
    print(message.content)
  #  await client.process_commands( message )
    if message.author == client.user:
        return
    if  message.content.lower() == ("!привет"):
        await message.channel.send('Hello!')
    if message.content.lower() == ('!time') :
        # сказать текущее время
        now = datetime.datetime.now()
        await message.channel.send("Now " + str(now.hour) + ":" + str(now.minute))
    elif message.content.lower() == ('!jokes') :
        # рассказать анекдот
        await message.channel.send("Мой разработчик не научил меня анекдотам ... Ха ха ха")
    elif message.content.lower() == ('!nlp_tag'):
        await message.channel.send("Insert text")
        await client.wait_for('message')
        print(message.content)
        doc = nlp(message.content)
        for entity in doc.ents:
            await message.channel.send(f"{entity.text} ({entity.label_})")
            print(f"{entity.text} ({entity.label_})")
    # elif cmd == 'NLP_Fact' :
        # print("Insert text")
        #text = input()
        #print("Insert word for fact")
       # word = input()
        #await client.send_message(message.channel,fact(text,word))
    #elif cmd == 'NLP_Placehold':
      #  global tik
        #global rep
       # print("Insert text, tok, replac")
       # text = input()
       #tik = input()
       # rep = input()
       # await client.send_message(message.channel, scrub(text))
    else:
      await message.channel.send(chat_tfidf(message.content))
      return


#@client.command(pass_context = True)
#async def  nlp_tag(ctx):
  #  await message.channel.send("Insert text")
    #await client.wait_for('message')
    #print(message.content)
#    doc = nlp(message.content)
  #  for entity in doc.ents:
    #    await message.channel.send(f"{entity.text} ({entity.label_})")
      #  print(f"{entity.text} ({entity.label_})")
    
client.run(DISCORD_BOT_TOKEN)
