from transformers import pipeline

# create pipeline for sentiment analysis
classification = pipeline('sentiment-analysis')
print(classification("I thoroughly enjoyed this movie!"))
print(classification("I did not understand anything in this movie."))

# let's try on difficult movie reviews
print(classification('''Although the movie had its flaws and some scenes were slow-paced, 
                I still found myself captivated by the strong performances and unique storyline. 
                The cinematography was stunning and added another level of depth to the film. Overall, 
                I would still recommend it for those looking for a thought-provoking movie experience.'''))

print(classification('''At first, I was intrigued by the premise and was excited to see where the story would go. 
                However, as the movie progressed, I became disappointed by the lackluster execution and 
                underdeveloped characters. Despite a few moments of promise, the movie ultimately fell 
                flat for me and failed to deliver on its potential. Not recommended.'''))

