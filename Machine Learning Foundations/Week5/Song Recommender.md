
# Building a song recommender


# Fire up GraphLab Create
(See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)


```python
import graphlab
```


```python
# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
```

# Load music data


```python
song_data = graphlab.SFrame('song_data.gl/')
```

# Explore data

Music data shows how many times a user listened to a song, as well as the details of the song.


```python
song_data.head()
```

## Showing the most popular songs in the dataset


```python
graphlab.canvas.set_target('ipynb')
```


```python
song_data['song'].show()
```


```python
len(song_data)
```

## Count number of unique users in the dataset


```python
users = song_data['user_id'].unique()
```


```python
len(users)
```

# Create a song recommender


```python
train_data,test_data = song_data.random_split(.8,seed=0)
```

## Simple popularity-based recommender


```python
popularity_model = graphlab.popularity_recommender.create(train_data,
                                                         user_id='user_id',
                                                         item_id='song')
```

### Use the popularity model to make some predictions

A popularity model makes the same prediction for all users, so provides no personalization.


```python
popularity_model.recommend(users=[users[0]])
```


```python
popularity_model.recommend(users=[users[1]])
```

## Build a song recommender with personalization

We now create a model that allows us to make personalized recommendations to each user. 


```python
personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id='user_id',
                                                                item_id='song')
```

### Applying the personalized model to make song recommendations

As you can see, different users get different recommendations now.


```python
personalized_model.recommend(users=[users[0]])
```


```python
personalized_model.recommend(users=[users[1]])
```

### We can also apply the model to find similar songs to any song in the dataset


```python
personalized_model.get_similar_items(['With Or Without You - U2'])
```


```python
personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])
```

# Quantitative comparison between the models

We now formally compare the popularity and the personalized models using precision-recall curves. 


```python
if graphlab.version[:3] >= "1.6":
    model_performance = graphlab.compare(test_data, [popularity_model, personalized_model], user_sample=0.05)
    graphlab.show_comparison(model_performance,[popularity_model, personalized_model])
else:
    %matplotlib inline
    model_performance = graphlab.recommender.util.compare_models(test_data, [popularity_model, personalized_model], user_sample=.05)
```

The curve shows that the personalized model provides much better performance. 


```python

```
