# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:02:01 2020

@author: 地三仙
"""

import tensorflow as tf  #  assert sys.modules[modname] is not old_mod 测试中反复运行会报错 
import os
print(" tensorflow version:{}".format(tf.__version__))
import re

def remove_html(text):
    r=re.compile(r'<[^>]+>')
    return r.sub('',text)



def file_read(file_type):
    """
    根据训练、测试类型读取不同文件夹下的数据集
    """
    path = "./data/aclImdb_v1/"
    file_list = []
    pos_file = path + file_type + '/pos/'
    for f in os.listdir(pos_file):
        file_list.append(pos_file + f)
    neg_file = path + file_type + '/neg/'
    for f in os.listdir(neg_file):
        file_list.append(neg_file + f)
    print("%s 数据集样本 %d 个文件！ " %(file_type, len(file_list)))
    # 情感标签 1为正面 0 为负面
    label_list = [1] * len(os.listdir(pos_file)) + [0] * len(os.listdir(pos_file))
    text = []
    for _f in file_list:
        with open(_f, encoding='utf8') as f:
            text += [remove_html(''.join(f.readlines()))]  # += extend 类似
                   
    return label_list, text
# 1.数据预处理
#用x表示label,y表示text里面的内容
x_train, y_train =  file_read('train')
x_test, y_test = file_read('test')

# 2.建立分词器
token=tf.keras.preprocessing.text.Tokenizer(num_words=2000) #建立一个有2000单词的字典
token.fit_on_texts(y_train) #读取所有的训练数据评论，按照单词在评论中出现的次数进行排序，前2000名会列入字典
#查看token读取多少文章
token.document_count

# 将评论数据转化为数字列表
train_seq=token.texts_to_sequences(y_train)
test_seq=token.texts_to_sequences(y_test)
#
# 让转换后的数字长度相同
#截长补短，让每一个数字列表长度都为100
_train=tf.keras.preprocessing.sequence.pad_sequences(train_seq,maxlen=100)
_test=tf.keras.preprocessing.sequence.pad_sequences(test_seq,maxlen=100)
len(_train[0])
len(train_seq[0])
print(y_train[0])
print(train_seq[0])
print(_train[0])


#
#加入嵌入层
#将数字列表转化为向量列表
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
Embedding = tf.keras.layers.Embedding

model_per = Sequential()
model_per.add(Embedding(output_dim=32, #将数字列表转换为32维的向量
                   input_dim=2000, #输入数据的维度是2000，因为之前建立的字典有2000个单词
                   input_length=100)) #数字列表的长度为100
model_per.add(Dropout(0.25))
#一、建立多层感知机模型
#加入平坦层
model_per.add(Flatten())

# 加入隐藏层
model_per.add(Dense(units=256,
               activation='relu'))
model_per.add(Dropout(0.35))

# 加入输出层
model_per.add(Dense(units=1,#输出层只有一个神经元，输出1表示正面评价，输出0表示负面评价
               activation='sigmoid'))

# 查看模型摘要
model_per.summary()

# 训练模型
model_per.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
# 数据必须是tensor 或者numpy 不能纯数组
#Failed to find data adapter that can handle input: <class 'numpy.ndarray'>,
# (<class 'list'> containing values of types {"<class 'int'>"})
train_history=model_per.fit(tf.constant(_train),tf.constant(x_train),batch_size=100,
                       epochs=10,verbose=2, validation_split=0.2) # 
#模型的fit函数有两个参数，shuffle用于将数据打乱，validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集

# 评估模型准确率
scores = model_per.evaluate(tf.constant(_test), tf.constant(x_test)) #第一个参数为feature,第二个参数为label


# 进行预测
predict = model_per.predict_classes(tf.constant(_test))
#转换成一维数组
predict = predict.reshape(-1)
type(predict[0])

_dict={1:'正面的评论',0:'负面的评论'}
def display(i):
    print(y_test[i])
    print('label真实值为:',_dict[x_test[i]],
         '预测结果为:',_dict[predict[i]])
display(0)

display(1)
# 完整函数
def review(input_text):
    input_seq=token.texts_to_sequences([input_text])
    pad_input_seq=tf.keras.preprocessing.sequence.pad_sequences(input_seq,maxlen=100)
    predict_result=model_per.predict_classes(tf.constant(pad_input_seq))
    print(_dict[predict_result[0][0]])

    
comment = '''Going into this movie, I had low expectations. I'd seen poor reviews, and I also kind of hate the idea of remaking animated films for no reason other than to make them live action, as if that's supposed to make them better some how. This movie pleasantly surprised me!
Beauty and the Beast is a fun, charming movie, that is a blast in many ways. The film very easy on the eyes! Every shot is colourful and beautifully crafted. The acting is also excellent. Dan Stevens is excellent. You can see him if you look closely at The Beast, but not so clearly that it pulls you out of the film. His performance is suitably over the top in anger, but also very charming. Emma Watson was fine, but to be honest, she was basically just playing Hermione, and I didn't get much of a character from her. She likes books, and she's feisty. That's basically all I got. For me, the one saving grace for her character, is you can see how much fun Emma Watson is having. I've heard interviews in which she's expressed how much she's always loved Belle as a character, and it shows.
The stand out for me was Lumieré, voiced by Ewan McGregor. He was hilarious, and over the top, and always fun! He lit up the screen (no pun intended) every time he showed up!
The only real gripes I have with the film are some questionable CGI with the Wolves and with a couple of The Beast's scenes, and some pacing issues. The film flows really well, to such an extent that in some scenes, the camera will dolly away from the character it's focusing on, and will pan across the countryside, and track to another, far away, with out cutting. This works really well, but a couple times, the film will just fade to black, and it's quite jarring. It happens like 3 or 4 times, but it's really noticeable, and took me out of the experience. Also, they added some stuff to the story that I don't want to spoil, but I don't think it worked on any level, story wise, or logically.
'''
# 惊奇发现：maxlen 120 -> 100 : neg -> pos
print(comment)

review(comment)


from tensorflow.keras.layers import LSTM
model_lstm=Sequential()

model_lstm.add(Embedding(output_dim=32,
                   input_dim=2000,
                   input_length=100))

model_lstm.add(Dropout(0.25))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(units=256,activation='relu'))
model_lstm.add(Dropout(0.25))
model_lstm.add(Dense(units=1,activation='sigmoid'))

model_lstm.summary()
model_lstm.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

train_history_lstm = model_lstm.fit(tf.constant(_train), tf.constant(x_train), batch_size=100,
                       epochs=10, verbose=2,
                       validation_split=0.2)

scores_lstm = model_lstm.evaluate(tf.constant(_test), tf.constant(x_test)) 