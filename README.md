# k_nearest_neighbor

## Requirement

- python>=3.7
- numpy

## Usage

```python
k = 3
knn = KNN(k, data)       # Make an instance.
cls = knn.run(sample)    # Make an inference. Belonging class will be output.
```

## Data

- Dataset must be given by list of dict
```python
data = [
    {"features": [8, 9], "class": "A"},
    {"features": [8, 4], "class": "A"},
    {"features": [4, 4], "class": "B"},
    {"features": [2, 4], "class": "B"},
    {"features": [7, 7], "class": "A"}
]
```
- Sample must be given by list
```python
sample = [3, 7]
```
