# NER Classifier

## Description  
This is a project under Edison Premium which contains the NER model used for classifying entities extraction from mail signature blocks

## Project Overview
### Architecture
After much comparison, we have settled with a Bi-lstm CRF neuralnetwork with word and character embeddings

### Components
- Train model and adjust hyperparameters: **bi_lstm.py**
- Convert model into tensorflow lite for mobile: **tf_converter.py**
- Evaluate model using test-set data: **batch_test.py**
- Quick-run to see model output: **tf_lite_invoke.py**

## Quick Run
Navigate to tf_lite_invoke.py

Run the script with the following code
```python
lite_model.set_sentence(sentence)
label = lite_model.analyze()
print(sentence, '：label is ', label)

```
Example Output
```text
Jordan McDonald ：label is  name
PO Box 7193 ：label is  loc
Gujarat, INDIA ：label is  loc
Legal, financial, technical translations ：label is  tit

```
## Training the Model

### Training Time
- Training the model locally is not recommended at data size processed is huge
- Instead, we train the model in a deployed ubtuntu VM
- Estimated time to train each model on the VM is ~= 4-6 hours

### Basic Train
Run the script in [bilstm_onefile/bi_lstm.py](https://github.com/yuanlida/nc/blob/master/bilstm_onefile/bi_lstm.py) directly
OR\
Run via Terminal
```python
python3 bilstm_onefile/bi_lstm.py
```
### VM 大机 Train
**Remember to change file directories and key to your own**
1. Turn on your VM server and SSH into it
```
aws ec2 start-instances --instance-ids i-029dfa9b95dba8117

ssh -i ~/.ssh/mykey ubuntu@34.212.42.106
```

2. Navigate to our nc project
```
cd nc 
```

3. Run the pre-defined shellscript via nohup and monitor progress
```
nohup sh ~/nc/run.sh >~/nc/script.py.log </dev/null 2>&1 &

tail -f ~/nc/script.py.log
```
4. Once training completes, copy 3 remote files to local: **bi-lilstm.tflite**, **charJson**, **wordJson** .
```
scp -i ~/.ssh/mykey -r ubuntu@34.212.42.106:~/nc/model/Bi-LSTM/bi-lstm.tflite ~/desktop/edison-ai/ner-tflite

scp -i ~/.ssh/mykey -r ubuntu@34.212.42.106:~/nc/model/Bi-LSTM/charJson ~/desktop/edison-ai/ner-tflite

scp -i ~/.ssh/mykey -r ubuntu@34.212.42.106:~/nc/model/Bi-LSTM/wordJson ~/desktop/edison-ai/ner-tflite
```
5. Cleanse the VM of these old files and data
```
sh clean.sh
```
6. Copy the contents of **bi-lilstm.tflite**, **charJson**, **wordJson** and paste them in [model/Bi-LSTM](https://github.com/yuanlida/nc/tree/master/model/Bi-LSTM)

7. Afterwhich, perform **Quick Run** as mentioned above

# NER Classifier

## Description  
This is a project under Edison Premium which contains the NER model used for classifying entities extraction from mail signature blocks

## Project Overview
### Architecture
After much comparison, we have settled with a Bi-lstm CRF neuralnetwork with word and character embeddings

### Components
- Train model and adjust hyperparameters: **bi_lstm.py**
- Convert model into tensorflow lite for mobile: **tf_converter.py**
- Evaluate model using test-set data: **batch_test.py**
- Quick-run to see model output: **tf_lite_invoke.py**

## Setup
```text
pip install -r requirements.txt
```

## Quick Run
Navigate to tf_lite_invoke.py

Run the script with the following code
```python
lite_model.set_sentence(sentence)
label = lite_model.analyze()
print(sentence, '：label is ', label)

```
Example Output
```text
Jordan McDonald ：label is  name
PO Box 7193 ：label is  loc
Gujarat, INDIA ：label is  loc
Legal, financial, technical translations ：label is  tit

```
## Training the Model

### Training Time
- Training the model locally is not recommended at data size processed is huge
- Instead, we train the model in a deployed ubtuntu VM
- Estimated time to train each model on the VM is ~= 4-6 hours

### Basic Train
Run the script in [bilstm_onefile/bi_lstm.py](https://github.com/yuanlida/nc/blob/master/bilstm_onefile/bi_lstm.py) directly
OR\
Run via Terminal
```python
python3 bilstm_onefile/bi_lstm.py
```
### VM 大机 Train
**Remember to change file directories and key to your own**
1. Turn on your VM server and SSH into it
```
aws ec2 start-instances --instance-ids i-029dfa9b95dba8117

ssh -i ~/.ssh/mykey ubuntu@34.212.42.106
```

2. Navigate to our nc project
```
cd nc 
```

3. Run the pre-defined shellscript via nohup and monitor progress
```
nohup sh ~/nc/run.sh >~/nc/script.py.log </dev/null 2>&1 &

tail -f ~/nc/script.py.log
```
4. Once training completes, copy 3 remote files to local: **bi-lilstm.tflite**, **charJson**, **wordJson** .
```
scp -i ~/.ssh/mykey -r ubuntu@34.212.42.106:~/nc/model/Bi-LSTM/bi-lstm.tflite ~/desktop/edison-ai/ner-tflite

scp -i ~/.ssh/mykey -r ubuntu@34.212.42.106:~/nc/model/Bi-LSTM/charJson ~/desktop/edison-ai/ner-tflite

scp -i ~/.ssh/mykey -r ubuntu@34.212.42.106:~/nc/model/Bi-LSTM/wordJson ~/desktop/edison-ai/ner-tflite
```
5. Cleanse the VM of these old files and data
```
sh clean.sh
```
6. Copy the contents of **bi-lilstm.tflite**, **charJson**, **wordJson** and paste them in [model/Bi-LSTM](https://github.com/yuanlida/nc/tree/master/model/Bi-LSTM)

7. Afterwhich, perform **Quick Run** as mentioned above

## Data
**Data files** are divided into:
- training set(90%)
- test set (10%)
- sample set (1%)

### Process
**Data Augmentation, generation, sorting:**
[process_data/jeff_work](https://github.com/yuanlida/nc/tree/master/process_data/jeff_work)

### Dataset
- [x] TIT: 
    - [osm 18](https://drive.google.com/drive/folders/1ebOcWihGh4HgBiqIfFE27hjZJ4-za85U?usp=sharing)
    - [oneTonline](https://www.kaggle.com/isanbel/all-occupations#All_Occupations.csv)
    - [Google jobs](https://www.kaggle.com/niyamatalmass/google-job-skills)
    - [Free Title_skills](https://drive.google.com/file/d/1Wy7UfYH-lpfoTKDm2rO4ytQekD63EFFg/view?usp=sharing), removed Spanish words
    - [NLPaug generation of job titles](https://drive.google.com/drive/folders/1_0RILbxsjIyjUaNK2KMotJNwny3bf1eF?usp=sharing)

- [x] TEL
    - follow US phone number format
    - [Edison data + artificial augmentation](https://drive.google.com/drive/folders/11HX1gI4CN1Ivi2OduvYDYoCnmotChIOO?usp=sharing) 
    - Added miscellaneous country number formats

- [x] ORG
    - [Changed to 7 mil company name corpus](https://drive.google.com/drive/folders/1mirZ0ln_0vkCr6bJWmE7pDOWTZFXSCBQ?usp=sharing), Included only 2,278,866 lines that are US based. 
    - [Crunchbase Companies data](https://drive.google.com/drive/folders/17m2uPym3k1HeBr_UZwlJU49lsLGYU9Tx?usp=sharing)

- [x] LOC
    - [added countries, states, streets](https://drive.google.com/drive/folders/1UZ1WqK5Ga-nRtha9i6IvGOT-XU-kRwzE?usp=sharing) 
    - [UK DATA](https://github.com/ideal-postcodes/postcodes.io/tree/master/data)
    - US KAGGLE open addresses data[DELETED]
    - [Added random places of interest](https://www.kaggle.com/ehallmar/points-of-interest-poi-database)
    - [Generated data](https://drive.google.com/drive/folders/1lGtY8Dy2z-o0KQzUdO25UZ0te5OLJ-qF?usp=sharing) with example address pattern formats are:
        - UK: street, town, county
        - US: street, city, state, postal
        
- [x] PER
    - Added single names
    - [Augment different name formats such as .D](https://drive.google.com/drive/folders/1yZMTBKW9XM-TAtlxFvm_vT5m_4UAAgCw?usp=sharing)
    - Edison email
    - [Baby names, USA names](https://drive.google.com/drive/folders/1gFa_LGndIC0qPp-1ICt1aKNnyCtvcsB6?usp=sharing)

## Model Evaluation Comparisons
View ![Performance](https://github.com/yuanlida/nc/blob/master/nc/performance)


## List of edge-cases / wrongly classified

### Should be LOC
- wildmoor ：label is  name

### Should be ORG

### Should be PER(name)
- Rob Record ：label is  loc
- Carla Roppo-Owczarek ：label is  org
- (BRIAN) HEXTER ：label is  org
- Senior Engineer - Projects & Services ：label is  org

### Should be TIT
- IT Support ：label is  org
- admin ：label is  loc