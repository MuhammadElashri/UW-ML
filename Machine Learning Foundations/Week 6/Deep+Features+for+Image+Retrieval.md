
# Building an image retrieval system with deep features


# Fire up GraphLab Create
(See [Getting Started with SFrames](../Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)


```python
import graphlab
```


```python
# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
```

# Load the CIFAR-10 dataset

We will use a popular benchmark dataset in computer vision called CIFAR-10.  

(We've reduced the data to just 4 categories = {'cat','bird','automobile','dog'}.)

This dataset is already split into a training set and test set. In this simple retrieval example, there is no notion of "testing", so we will only use the training data.


```python
image_train = graphlab.SFrame('image_train_data/')
```

# Computing deep features for our images

The two lines below allow us to compute deep features.  This computation takes a little while, so we have already computed them and saved the results as a column in the data you loaded. 

(Note that if you would like to compute such deep features and have a GPU on your machine, you should use the GPU enabled GraphLab Create, which will be significantly faster for this task.)


```python
# deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
# image_train['deep_features'] = deep_learning_model.extract_features(image_train)
```


```python
image_train.head()
```

# Train a nearest-neighbors model for retrieving images using deep features

We will now build a simple image retrieval system that finds the nearest neighbors for any image.


```python
knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],
                                             label='id')
```

# Use image retrieval model with deep features to find similar images

Let's find similar images to this cat picture.


```python
graphlab.canvas.set_target('ipynb')
cat = image_train[18:19]
cat['image'].show()
```


```python
knn_model.query(cat)
```

We are going to create a simple function to view the nearest neighbors to save typing:


```python
def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')
```


```python
cat_neighbors = get_images_from_ids(knn_model.query(cat))
```


```python
cat_neighbors['image'].show()
```

Very cool results showing similar cats.

## Finding similar images to a car


```python
car = image_train[8:9]
car['image'].show()
```


```python
get_images_from_ids(knn_model.query(car))['image'].show()
```

# Just for fun, let's create a lambda to find and show nearest neighbor images


```python
show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()
```


```python
show_neighbors(8)
```


```python
show_neighbors(26)
```


```python

```
