import pandas as pd; import re,textwrap,pickle,os; from transformers import pipeline; from tqdm import tqdm
# create pipeline for sentiment analysis
classification = pipeline('sentiment-analysis')

def cleanup(text):
    text=re.sub(r"[^a-z0-9\.\?\! ]", " ",text.lower())
    text=re.sub(r'\.+','.',text)
    text=' '.join(text.split()).strip()
    return text
# Load the dataset
data = pd.read_csv('sorted_data_acl.csv', keep_default_na=False,delimiter=',')

labels=data['sentiment'].apply(lambda x: 0 if x[0] == 'p' else 1)

# Convert rating to human language sentence that lang transformer can pay attention to
data['rating']=data['rating'].apply(lambda x:  'user rating '+str(int(x)))
# Convert usefull to human language sentence that lang transformer can pay attention to
data['helpful']=data['helpful'].apply(lambda x:  'deemed helpful by '+x if len(x) else x)
# Merge to one one review text but keep just lowercase alnum and cleanup \n
# keep !? those carry strong emotions too
# example train row text: user rating 1. deemed helpful by 4 of 9. horrible book horrible. the..
texts=(data['rating']+'. '+data['helpful']+'. '+data['title']).apply(cleanup)+'; '+data['review_text'].apply(cleanup)

results=[]
if os.path.exists('results.pickle'):
    with open('results.pickle','rb') as f:
        results= pickle.load(f)
else:
    for text in tqdm(texts):

        short_summary_and_numbers,long_review_text=text.split(';')
        paragraphs = textwrap.wrap(long_review_text, 450)

        subresults=[]
        #model cant process more then 450 tokens at once so we need to work in chunks and avg results
        for paragraph in paragraphs:
            subresults+=classification(short_summary_and_numbers.strip()+' '+paragraph.strip())

        # Calculate the average rating
        avg_rating = sum([subresult['score'] for subresult in subresults]) / len(subresults)

        # Calculate the average sentiment
        mp={'NEGATIVE':0,'POSITIVE':1}
        avg_sentiment = round(sum([mp[subresult['label']] for subresult in subresults]) / len(subresults))
        
        results+=tuple(avg_sentiment,avg_rating)

    with open('results.pickle','wb') as f:
        pickle.dump(results,f)

labels = [1 - x for x in labels]

possibly_detected_outliers=[]
correct_predictions=0
pred_labels=[]
for i,p in enumerate(results):
    possibly_detected_outliers.append([results[i][0],labels[i],texts[i]])
    pred_labels.append(results[i][0])
    result = int(labels[i] == results[i][0])
    if not result:
        possibly_detected_outliers.append(i)
    correct_predictions+=result

accuracy = correct_predictions / len(labels)

print(f'Accuracy: {accuracy}')

# print confusion matrix 
from sklearn.metrics import confusion_matrix

# Print Confusion Matrix
conf_matrix = confusion_matrix(labels, pred_labels)
print('\nConfusion Matrix:')
print(conf_matrix)


