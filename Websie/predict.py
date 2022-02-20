import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')


def cleanString(incomingString):
    newstring = incomingString
    newstring = newstring.replace("!","")
    newstring = newstring.replace("@","")
    newstring = newstring.replace("#","")
    newstring = newstring.replace("$","")
    newstring = newstring.replace("%","")
    newstring = newstring.replace("^","")
    newstring = newstring.replace("&","and")
    newstring = newstring.replace("*","")
    newstring = newstring.replace("(","")
    newstring = newstring.replace(")","")
    newstring = newstring.replace("+","")
    newstring = newstring.replace("=","")
    newstring = newstring.replace("?"," ")
    newstring = newstring.replace("\'","")
    newstring = newstring.replace("\"","")
    newstring = newstring.replace("'","")
    newstring = newstring.replace("'m","am")
    newstring = newstring.replace("}","")
    newstring = newstring.replace("[","")
    newstring = newstring.replace("]","")
    newstring = newstring.replace("<","")
    newstring = newstring.replace(">","")
    newstring = newstring.replace("~","")
    newstring = newstring.replace("`","")
    newstring = newstring.replace(":","")
    newstring = newstring.replace(";","")
    newstring = newstring.replace("|","")
    newstring = newstring.replace("\\","")
    newstring = newstring.replace("/","") 
    newstring = newstring.replace("0","")
    newstring = newstring.replace("1","")
    newstring = newstring.replace("2","")
    newstring = newstring.replace("3","")
    newstring = newstring.replace("4","")
    newstring = newstring.replace("5","")
    newstring = newstring.replace("6","")
    newstring = newstring.replace("7","")
    newstring = newstring.replace("8","")
    newstring = newstring.replace("9","")  
    newstring = newstring.replace(".","")
    newstring = newstring.replace(",","")
    return newstring

def predict(sen):
    pickle_in = open("classifier.pkl","rb")
    classifier=pickle.load(pickle_in)
    pickle_in.close()
    text_ = cleanString(sen)
    text_tokens = word_tokenize(text_)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    text = (" ").join(tokens_without_sw)
    pickle_in = open("model.pkl","rb")
    model=pickle.load(pickle_in)
    pickle_in.close()
    vec = model.texts_to_matrix([text], mode='count')
    prediction = classifier.predict(vec)
    return prediction

if __name__ == "__main__":
    print(predict("Congrats, you won 1000$"))


