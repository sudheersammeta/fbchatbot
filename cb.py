
# coding: utf-8

# In[1]:

from __future__ import print_function
import sys
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import numpy as np
import math

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

tf.reset_default_graph()

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


x_train = np.array(train_data[0])
y_train = np.array(train_data['class'])
x_test = np.array(train_data[0])
y_test = np.array(train_data['class'])
print (x_train)
y_train = np.array(y_train).reshape([-1, 1])
print (y_train)


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(x_train)
x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)
input_size =  len(vect.get_feature_names())

query = ["Mickey Mouse"]
msg = vect.transform(query)
msg =  msg.toarray()

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

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

prediction_array  = 0;
saver = tf.train.Saver()
sess =  tf.Session()
  # Restore variables from disk.
saver.restore(sess, "./ANN_checkpoints/ANN")
while True:
    input_text = [_get_user_input()]
    text_string = input_text[0]
    print(input_text)
    input_text = vect.transform(input_text).toarray()
    pred_arr = np.array(sess.run(prediction, feed_dict = {X:input_text}))
    print (pred_arr)
    prediction_array = pred_arr
    max_index = np.argmax(pred_arr)
    probability = pred_arr[0][max_index]
    print (probability)
    tf.reset_default_graph()
    if probability > 0.5:
        print (np.random.choice(responses[class_labels[np.argmax(prediction_array)]]))
    else:
        if (probability>.27):
            print('Did you mean ' + class_labels[np.argmax(prediction_array)] +'?')
            print('Enter y or n \n')
            yrn = _get_user_input()            
            print(yrn)
            if(yrn == 'y\n'):
                print (np.random.choice(responses[class_labels[np.argmax(prediction_array)]]))
            else:
                print ("Fallback")
                from cmpe297_bot import chat_2
                print (chat_2(text_string))
        else:
                print ("Fallback")
                from cmpe297_bot import chat_2
                print (chat_2(text_string))
            

      