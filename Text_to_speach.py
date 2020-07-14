from gtts import gTTS 
import os 
audio_file = 'D:\\Project\\Dissertation\\Audio\\Welcome.mp3'
mytext = 'Welcome to RSS Navigation App!'
  
# Language in which you want to convert 
language = 'en'
  
# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
myobj = gTTS(text=mytext, lang=language, slow=False) 
  
# Saving the converted audio in a mp3 file named 
# welcome  
myobj.save(audio_file) 
  
# Playing the converted file 
#os.system("mpg321 welcome.mp3") 