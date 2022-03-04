from googletrans import Translator
translator=Translator()
s=translator.translate('i love you', dest='sv')
print(s.text)
