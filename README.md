# nc  
完整模式用train训练，用test验证  
测试模式用test训练，用test验证  

1. 适配edison数据  搞定  
2. 多分类问题 搞定    
3. 训练测试一体化  搞定
4. 随机矩阵使用 搞定
5. character + word  改变sequece的网络结构使其之能够实现预测功能。 搞定
6. 选择最佳方案  @Jeffery
7. 保存为TFLite  搞定。
8. glove引入，解决大小写问题.推迟   

1.  一部分准确率低的原因是随机矩阵维度太低。  
2.  维度低可以加快迭代速度。  
3.  Jeffery做模型测试，看模型大小，和精确度。  
4.  Dalio修改charcnn模型后，移植bi_lstm_wc.  
5.  如果是数字，在word内提取出来，在char内是不是提取，看精度决定。 by Jerffery
