# Progress Documentation

## Evaluation metrics: 
- Primary: Recall (greater focus on ensuring False Negatives are accounted for)
- Secondary: Accuracy, Precision 

## Legend
- w = weights of word embedding layer
- c = weights of character embedding layer
- hw = weights of hidden layer of word embedding
- hc = weights of hidden layer of character embedding
- lr = learning rate
- l2 = L2 regularisation value
- training set = 90% of full data sample used to train model
- dev set = 10% of full data sample used to evaluate model
- test set = unseen edison data from QA used to evaluate model

## Important Notes
- Scores Displayed below are in recall
- Models are evaluated against a dev set (10% from the full sample, 90% used for training)
- Good performance of dev set does not ensure the same performance on test set
- QA data has helped to capture edge cases of model, and provided direction to resolve conflicts in PER, ORG, LOC, TIT data 

## Edison Premium Deployed model
### M76
- Data changes
- Epoch = 4
- L2 = 1
- lr=0.01
- W,c = 50,100
- Hw,hc = 50,100
![m76](https://github.com/yuanlida/nc/blob/master/performance/m76.png)

-----------------

## Older Versions
![m10-11](https://github.com/yuanlida/nc/blob/master/performance/m10-11.png)


![m1-9](https://github.com/yuanlida/nc/blob/master/performance/m1-9.png)

### M22:
- ADDED QA
- ADDED Augmented TIT
- 10 mistakes till #65

### M23
- Train/per/line 219605 after Is swap_per
- 100,50

### M24: 
- L2 = 0.01
- Everything else normal 
50,50 . 5,50
￼
### M27[OK]
- Dim word, char = 50,70
- Hw, hc = 50, 50

### M33
- Generated new names
- Word,char = 50,100
- Hw,hc = 50,100

### M37
- [loc inaccuracies, 7 errors]
- Clean some loc,
- Word, char = 100,50
- Hw, hc = 100,50

### M42
[GOOD]
- Embedding size = 50
- Word, char = 100,50
- LR =0.001
- L2 = 1

### M43
- Cap names bad
- Change all data to .lower()

### M51
- L2 =1 
- Lr = 0.001
- Embedding = 50
- W,c = 100, 50
- Hw, hc = 100, 50
- M50-51 performed terribly in compromising certain names, tit for org etc.
- Reduced noise in TIT to prevent false positives of TIT

### M52
- 10k TIT
- Embedding = 50
- W,c = 100, 100
- Hw, hc = 100, 100
￼
￼
### M53
- Model size reduced
- Embedding = 50
- W,c = 100, 100
- Hw, hc = 100, 100
￼
## M67
- [changed tIT -org errors]
- Epoch = 5
- Embedding = 50
- W,c =  100,50
- Hw, hc = 120,50
- Better in TIT
￼
###  M68
- [changed tIT -org errors]
- Epoch = 5
- Embedding = 50
- W,c =  100,50
- Hw, hc = 100,50
- Better in name
￼
### M69
- Epoch = 5
- Embedding = 50
- W,c =  100,50
- Hw, hc = 150,30
- Too little characters result in misclassification of TIT false positives of random phrases

### M70
- [cleaned names]
- Epoch = 5
- Embedding = 50
- W,c =  50,100
- Hw, hc = 50,100
- Dev/ valida/nction set done in training 
![m70](https://github.com/yuanlida/nc/blob/master/performance/m70.png)


### M71
[ cleaned names]
- Epoch = 5
- Embedding = 50
- W,c =  120, 50
- Hw, hc = 120,50
— concluded that char>word performs better or comparatively good as well

### M73
- [cleaned org]
- Epoch = 5
- Embedding = 50
- W,c =  70, 80
- Hw, hc = 70, 80
￼
￼
### M78
- Data changes
- Epoch = 5
- L2 = 1
- lr=0.01
- W,c = 100,70
- Hw,hc = 100,70
- Score: 10
![m78](https://github.com/yuanlida/nc/blob/master/performance/m78.png)
