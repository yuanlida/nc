# nc  
完整模式用train训练，用test验证  
测试模式用test训练，用test验证  

1. 适配edison数据  搞定
2. 多分类问题 搞定 charcnn多分类精度很低  
3. 训练测试一体化  搞定
4. 随机矩阵使用 搞定
5. character + word  改变sequece的网络结构使其之能够实现预测功能。 搞定
6. 选择最佳方案  @Jeffery
7. 保存为TFLite  今晚搞定。
8. glove引入，解决大小写问题  

一部分准确率低的原因是随机矩阵维度太低。  
维度低可以加快迭代速度。  