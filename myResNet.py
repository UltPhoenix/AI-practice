import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import layers,Sequential,regularizers,datasets,metrics
from tensorflow_core.python.keras.api._v2.keras import optimizers
import numpy as np 
class BasicBlock(layers.Layer):
    def __init__(self,filter_num,stride=1,kind='res'):
        super(BasicBlock,self).__init__()
        self.kind=kind
        if kind=='res':
            self.conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
            self.bn1=layers.BatchNormalization()
            self.relu=layers.ReLU()
            self.conv2=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
            self.bn2=layers.BatchNormalization()
            if stride!=1:
                self.downsample=Sequential()
                self.downsample.add(layers.Conv2D(filter_num,(1,1),strides=stride))
            else:
                self.downsample= lambda x:x
        elif kind=='Dense':
            self.conv1=layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
            self.bn1=layers.BatchNormalization()
            self.relu=layers.ReLU()
            self.conv2=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
            self.bn2=layers.BatchNormalization()
            self.conv3=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
            self.bn3=layers.BatchNormalization()
            self.conv4=layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
            self.bn4=layers.BatchNormalization()
            if stride!=1:
                self.downsample13=Sequential()
                self.downsample13.add(layers.Conv2D(filter_num,(1,1),strides=stride))
                self.downsample14=Sequential()
                self.downsample14.add(layers.Conv2D(filter_num,(1,1),strides=stride))
                self.downsample15=Sequential()
                self.downsample15.add(layers.Conv2D(filter_num,(1,1),strides=stride))
                self.downsample=lambda x:x
            else:
                self.downsample= lambda x:x
    def call(self,inputs,training=None):
        if self.kind=='res':
            out=self.conv1(inputs)
            out=self.bn1(out)
            out=self.relu(out)
            out=self.conv2(out)
            out=self.bn2(out)
            identity=self.downsample(inputs)
            output=layers.add([out,identity])
            output=tf.nn.relu(output)
        elif self.kind=='Dense':
            mid1=self.conv1(inputs)
            mid1=self.bn1(mid1)
            mid1=self.relu(mid1)
            mid2=self.conv2(mid1)
            mid2=self.bn2(mid2)
            identity1=self.downsample13(inputs)
            mid2=layers.add([mid2,identity1])
            mid2=self.relu(mid2)
            mid3=self.conv3(mid2)
            mid3=self.bn3(mid3)
            mid3=self.bn3(mid3)
            identity=self.downsample14(inputs)
            mid3=layers.add([mid3,identity,mid1])
            mid3=self.relu(mid3)
            mid4=self.conv4(inputs)
            mid4=self.bn4(mid1)
            identity=self.downsample15(inputs)
            mid4=layers.add([mid4,mid2,mid1,identity])
            output=self.relu(mid4)
        return output

class ResNet(keras.Model):
    def __init__(self,_lambda,layer_dims,num_classes=10,kind='res'):
        super(ResNet,self).__init__()
        self.stem=Sequential([layers.Conv2D(64,(3,3),strides=(1,1)),layers.BatchNormalization(),layers.ReLU(),layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')])
        self.layer1=self.bulid_block(64,layer_dims[0],kind=kind)
        self.layer2=self.bulid_block(128,layer_dims[1],stride=2,kind=kind)
        self.layer3=self.bulid_block(256,layer_dims[2],stride=2,kind=kind)
        self.layer4=self.bulid_block(512,layer_dims[3],stride=2,kind=kind)
        self.avg_pool=layers.GlobalAveragePooling2D()
        self.fcm=layers.Dense(1000,activation='relu',kernel_regularizer=regularizers.l2(_lambda))
        self.fc=layers.Dense(num_classes)
    def bulid_block(self,filter_num,blocks,stride=1,kind='res'):
        resblocks=Sequential()
        resblocks.add(BasicBlock(filter_num,stride,kind))
        for _ in range(1,blocks):
            resblocks.add(BasicBlock(filter_num,stride=1))
        return resblocks
    def call(self,inputs,training=None):
        x=self.stem(inputs)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avg_pool(x)
        x=self.fcm(x)
        x=self.fc(x)

        return x 

def resnet18(lam):
    return ResNet(lam,[2,2,2,2])
def resnet34(lam):
    return ResNet(lam,[3,4,6,3])

def preprocess(x,y):
    x=2*tf.cast(x,dtype=tf.float32)/255-1
    y=tf.cast(y,dtype=tf.int32)
    return x,y

acc_meter=metrics.Accuracy()
model=resnet34(0.00097)
model(tf.random.normal([512,32,32,3]))
acc=[]
optimizer=optimizers.RMSprop(0.001)

(x,y),(x_test,y_test)=datasets.cifar10.load_data()
y=tf.squeeze(y,axis=1) 
y_test=tf.squeeze(y_test,axis=1)
print(x.shape,y.shape,x_test.shape,y_test.shape)
train_db=tf.data.Dataset.from_tensor_slices((x,y))
train_db=train_db.shuffle(1000).map(preprocess).batch(512)
test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db=test_db.map(preprocess).batch(512)


for epoch in range(50):
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            logits=model(x)
            pred=tf.nn.softmax(logits,axis=-1)
            pred=tf.argmax(pred,axis=1)
            pred=tf.cast(pred,dtype=tf.int32)
            y_onehot=tf.one_hot(y,depth=10)
            acc_meter.update_state(y,pred)
            loss=tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
            loss=tf.reduce_mean(loss)
            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            acc.append(acc_meter.result().numpy())
            print(step,"Evaluate Acc:",acc_meter.result().numpy())
            acc_meter.reset_states()
tf.saved_model.save(model,'resnet34_test')
print("model saved")

