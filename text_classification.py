
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from sklearn import metrics
import numpy as np
import math


# In[2]:


class_labels = ['About Class', 'Exam Details', 'Greetings', 'Project Details', 'Syllabus', 'Valediction', 'Professor Name', 'Office Hours', 'Office Location', 'Lecture Timings', 'Lecture Location', 'Class Location']
dataframe = pd.read_table("./JSGF_Files/professorName.txt", header=None)
dataframe['class'] = 6
train_data = dataframe
#print train_data


# In[3]:


dataframe = pd.read_table("./JSGF_Files/officeHours.txt", header=None)
dataframe['class'] = 7 #'Office Hours'#1
train_data = train_data.append(dataframe)


# In[4]:


dataframe = pd.read_table("./JSGF_Files/officeLocation.txt", header=None)
dataframe['class'] = 8 #'Office Location'#2
train_data = train_data.append(dataframe)


# In[5]:


dataframe = pd.read_table("./JSGF_Files/lectureTimings.txt", header=None)
dataframe['class'] = 9 #'Lecture Timings'#3
train_data = train_data.append(dataframe)


# In[6]:


dataframe = pd.read_table("./JSGF_Files/lectureLocation.txt", header=None)
dataframe['class'] = 10 #'Lecture Location'#4
train_data = train_data.append(dataframe)


# In[7]:


dataframe = pd.read_table("./JSGF_Files/classLocation.txt", header=None)
dataframe['class'] = 11 #'Class Location'#5
train_data = train_data.append(dataframe)


# In[8]:


dataframe = pd.read_table("./JSGF_Files/aboutClass.txt", header=None)
dataframe['class'] = 0
train_data = train_data.append(dataframe)


# In[9]:


dataframe = pd.read_table("./JSGF_Files/examDetails.txt", header=None)
dataframe['class'] = 1
train_data = train_data.append(dataframe)


# In[10]:


dataframe = pd.read_table("./JSGF_Files/greetings.txt", header=None)
dataframe['class'] = 2
train_data = train_data.append(dataframe)


# In[11]:


dataframe = pd.read_table("./JSGF_Files/projectDetails.txt", header=None)
dataframe['class'] = 3
train_data = train_data.append(dataframe)


# In[12]:


dataframe = pd.read_table("./JSGF_Files/syllabus.txt", header=None)
dataframe['class'] = 4
train_data = train_data.append(dataframe)


# In[13]:


dataframe = pd.read_table("./JSGF_Files/valediction.txt", header=None)
dataframe['class'] = 5
train_data = train_data.append(dataframe)


# In[14]:


print train_data


# In[15]:


x_train = np.array(train_data[0])
y_train = np.array(train_data['class'])
x_test = np.array(train_data[0])
y_test = np.array(train_data['class'])
print x_train
y_train = np.array(y_train).reshape([-1, 1])
print y_train


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(x_train)
x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)
input_size =  len(vect.get_feature_names())
#print x_train_dtm
simple_train_dtm = vect.transform(x_train)
#simple_train_dtm
#pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
x_cnn = np.array(simple_train_dtm.toarray(), np.float32)
print x_cnn[0]
#y_cnn = np.array(tf.one_hot(y_train,6))
#print y_cnn


# In[17]:


query = ["When is the class"]
msg = vect.transform(query)
msg =  msg.toarray()
print msg


# # input placeholders
# X = tf.placeholder(tf.float32, [None, input_size])
# X_img = tf.reshape(X, [-1,1,input_size,1]) 
# y = tf.placeholder(tf.int32, [None, 1])
# Y = tf.one_hot(y,6)
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# L1 = tf.nn.conv2d(X_img, W1,strides=[1, 1, 1, 1], padding='SAME')
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
# W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
# W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 6], initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([6]))
# hypothesis = tf.matmul(L2, W3) + b
# # define cost/loss & optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

# In[18]:


n_class = 12;
X = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.int32, [None, 1])
Y = tf.one_hot(y, n_class)
W1 = tf.Variable(tf.truncated_normal([input_size, 256], stddev=math.sqrt(2.0/(28*28))))
b1 = tf.Variable(tf.zeros([256]))
W2 = tf.Variable(tf.truncated_normal([256, 64], stddev=math.sqrt(2.0/(256))))
b2 = tf.Variable(tf.zeros([64]))
W3 = tf.Variable(tf.truncated_normal([64, n_class], stddev=math.sqrt(2.0/(64))))
b3 = tf.Variable(tf.zeros([n_class]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
hypothesis = tf.nn.relu(tf.matmul(layer2, W3) + b3)
prediction = tf.nn.softmax(hypothesis)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
cost_summ = tf.summary.scalar("cost", cost)
summary = tf.summary.merge_all()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.03).minimize(cost)
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# In[19]:


# initialize
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# train my model
print('Learning stared. It takes sometime.')
for epoch in range(80):
    avg_cost = 0
    #sess.run(y_cnn)
    #y_cnn = np.array(tf.one_hot(y_train,6))
    #total_batch = int(mnist.train.num_examples / batch_size)   
    #for i in range(4064):
    #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #for i in range(4064):
    #feed_dict = {X: np.array(x_cnn).reshape([1, 32]), Y: np.array(y_cnn[i]).reshape([-1,1])}
    feed_dict = {X: x_cnn.reshape([-1, input_size]), y: np.array(y_train)}
    s,c, _, = sess.run([summary, cost, optimizer],feed_dict=feed_dict)       
    avg_cost += c / 4064
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
save_path = saver.save(sess,"./ann_checkpoints/ann_model")

# In[20]:


pred_arr = np.array(sess.run(prediction, feed_dict = {X:msg}))
print pred_arr
max_index = np.argmax(pred_arr)
probability = pred_arr[0][max_index]
print probability


# In[21]:


responses = {'About Class':['This class deals with Deep Neural Networks and their applications to various problems such as speech recognition, image segmentation, and natural language processing. In addition, It covers underlying theory, the range of applications to which it has been applied, and hands on labs and projects to learn about Deep Learning'  ],
             'Exam Details':['One Midterm exam (On OCT 17,2017) and One Final exam (on Dec 14, 2017), With Midteam weighting 10% and Finals weighting 20% of total Grade','Midterm is on Oct 17 and Final is on Dec 14'],
             'Greetings':['Hi','Hi, I am CMPE297-Team Tensor\'s Chat Bot', 'Hi there', 'Hello','Hello, Welcome', 'Hello, I am CMPE297-Team Tensor\'s Chat Bot'],
             'Project Details':["You can Download the project details from https://sjsu.instructure.com/files/48338070/download?download_frd=1","The details about the project is available at https://sjsu.instructure.com/files/48338070/download?download_frd=1","Go to https://sjsu.instructure.com/files/48338070/download?download_frd=1 to view project details"],
             'Syllabus':["You can view/Download Syllabus of CMPE 297- Deep Learning Class from 'https://sjsu.instructure.com/files/47951447/download?download_frd=1","You can Download the Syllabus from https://sjsu.instructure.com/files/47951447/download?download_frd=1", "Go to https://sjsu.instructure.com/files/47951447/download?download_frd=1 to Download the syallbus for CMPE 297- Deep Learning Class"],
             'Valediction':['Bye', 'Good Bye', 'See you', 'Thanks for chatting with me, Good Bye', 'Ciao'],
             'Professor Name':['Its Prof. Simon Shim'],
             'Office Hours':['Every Mon 2:30PM to 4PM','Every Monday from 2:30PM to 4:00PM', 'Mondays 2:30 - 4 PM'],
             'Office Location':['It\'s in ENGR 269','It\'s in Engineering Building 269', 'It\'s in Engineering Building, Room - 269'],
             'Lecture Timings':['Every Tue 3PM to 5:45PM','Every Tuesday from 3PM to 5:45PM', 'Tuesdays 3 - 5:45PM'],
             'Lecture Location':['It\'s in HB 407', 'It\'s in Health Building 407', 'It\'s in Health Building, Room - 407'],
             }


# In[22]:


if probability > 0.5:
    print np.random.choice(responses[class_labels[np.argmax(pred_arr)]])
else:
    print "Fallback"
    from cmpe297_bot import chat_2
    print chat_2(query)






