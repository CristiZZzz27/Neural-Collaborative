# Neural Collaborative Filtering with MovieLens 

## Dependency

```java
pytorch >= 1.12.0
python >= 3.8
scipy >= 1.7.1
numpy >= 1.20.3
```





## Running the code



### local train (for more than 4h on P100)

```
python main.py --epoch 30 --batch 256 --factor 8 --model NeuMF  --topk 10 --file_size 1m --use_pretrain False --save True
```



### local test

It can random choose 100 testdata to produce

```
python testdata.py
```

or you can just run the testdata.py



### online train

```
!pip3 install mxnet-mkl==1.6.0 numpy==1.23.1 #(for colab and kaggle)
```



```
!git clone -b master https://github.com/CristiZZzz27/Neural-Collaborative.git
```



```python
cd Neural-Collaborative
```



```python
!python main.py --epoch 30 --batch 256 --factor 8 --model NeuMF  --topk 10 --file_size 1m --use_pretrain False --save True
```

There are no online test  because the pretrain file(.pth) is too big to upload to github .
