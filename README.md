# NER Classifier

## Description  
This is a project under Edison Premium which contains the NER model used for classifying entities extraction from mail signature blocks

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

## Project Overview
### Architecture
After much comparison, we have settled with a Bi-lstm CRF neuralnetwork with word and character embeddings

### Components
- Train model and adjust hyperparameters: **bi_lstm.py**
- Convert model into tensorflow lite for mobile: **tf_converter.py**
- Evaluate model performance: **tf_lite_invoke.py**

## Model Comparisons
### Best Performance:
Bilstm-onefile78