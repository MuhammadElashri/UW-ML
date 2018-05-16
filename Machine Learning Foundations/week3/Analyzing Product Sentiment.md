
# Predicting sentiment from product reviews

# Fire up GraphLab Create
(See [Getting Started with SFrames](/notebooks/Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)


```python
import graphlab
```


```python
# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
```

# Read some product review data

Loading reviews for a set of baby products. 


```python
products = graphlab.SFrame('amazon_baby.gl/')
```

# Let's explore this data together

Data includes the product name, the review text and the rating of the review. 


```python
products.head()
```

# Build the word count vector for each review


```python
products['word_count'] = graphlab.text_analytics.count_words(products['review'])
```


```python
products.head()
```


```python
graphlab.canvas.set_target('ipynb')
```


```python
products['name'].show()
```

# Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'


```python
giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
```


```python
len(giraffe_reviews)
```


```python
giraffe_reviews['rating'].show(view='Categorical')
```

# Build a sentiment classifier


```python
products['rating'].show(view='Categorical')
```

## Define what's a positive and a negative sentiment

We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   


```python
# ignore all 3* reviews
products = products[products['rating'] != 3]
```


```python
# positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4
```


```python
products.head()
```

## Let's train the sentiment classifier


```python
train_data,test_data = products.random_split(.8, seed=0)
```


```python
sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)
```

# Evaluate the sentiment model


```python
sentiment_model.evaluate(test_data, metric='roc_curve')
```


```python
sentiment_model.show(view='Evaluation')
```

# Applying the learned model to understand sentiment for Giraffe


```python
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')
```


```python
giraffe_reviews.head()
```

## Sort the reviews based on the predicted sentiment and explore


```python
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
```


```python
giraffe_reviews.head()
```

## Most positive reviews for the giraffe


```python
giraffe_reviews[0]['review']
```


```python
giraffe_reviews[1]['review']
```

## Show most negative reviews for giraffe


```python
giraffe_reviews[-1]['review']
```


```python
giraffe_reviews[-2]['review']
```


```python

```
