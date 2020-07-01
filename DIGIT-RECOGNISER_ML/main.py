from modelnetwork import *
from modellayers import DenseLayer, ActivationLayer
import time
import six.moves.cPickle as pickle
import gzip
import numpy as np
import csv
import os.path
from os import path


INPUT = 784


def to_categorical(j):
    c = np.zeros((10, 1))
    j = int(j[0])
    c[j] = 1.0
    return c

def group_data(data):
    features = [np.reshape(x, (INPUT, 1)) for x in data[0]]
    labels = [to_categorical(y) for y in data[1]]
    return list(zip(features, labels))

def main(tr_img_csv_path=None, tr_label_csv_path=None, test_img_csv_path=None):
    dummy_test_label = False
    
    start_time = time.time()
    
    print(tr_img_csv_path, tr_label_csv_path, test_img_csv_path)
    if tr_img_csv_path is None:
        tr_img_csv_path = "train_image.csv"
    if tr_label_csv_path is None:
        tr_label_csv_path = "train_label.csv"
    if test_img_csv_path is None: 
        test_img_csv_path = "test_image.csv"
    
    #tag_vocareum_dataset_loading_start VOCAREUM (60K, 0, 10K)
    with open(tr_img_csv_path, 'r') as f:
       tr_image = list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    with open(tr_label_csv_path, 'r') as f:
       tr_label = list(csv.reader(f, delimiter=','))
    
    tr_data = (tr_image, tr_label)
    training_data = group_data(tr_data)
    print("Training data shape: ", len(training_data))
    
    with open(test_img_csv_path, 'r') as f:
       test_image = list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    if path.exists("test_label.csv"):
        with open('test_label.csv', 'r') as f:
           test_label = list(csv.reader(f, delimiter=','))
    else:
        print("DUMMY TEST LABEL AS test_label is not provided, so simply write to test_predictions.csv")
        test_label = tr_label # DUMMY test_label
        dummy_test_label = True
    
    tst_data = (test_image, test_label)
    test_data = group_data(tst_data)
    print("Test data shape: ", len(test_data))
    #tag_vocareum_dataset_loading_end    
    

    model = Sequential() 
    #Weight Init: XavierUniform, HeNormal
    model.add(DenseLayer(784, 64, "H1", "HeNormal")) 
    model.add(ActivationLayer(64, "A1", "sigmoid"))
    model.add(DenseLayer(64, 32, "H2", "HeNormal"))
    model.add(ActivationLayer(32, "A2", "sigmoid"))
    model.add(DenseLayer(32, 10, "OL", "HeNormal"))
    model.add(ActivationLayer(10, "A3", "softmax"))   # SOFTMAX # LOSS: CROSS ENTROPY
 

    print("Time taken before the training start : ", time.time() - start_time)
    
    train_start_time = time.time()
    model.fit(training_data, num_iter=60, learning_rate= 0.02, 
              mini_batch_size=128) #test_data=validate_data)
    
    
    print("Training Execution time : ", time.time() - train_start_time)
    eval_write_time = time.time()
    print("Training is done, now predict on test_data and write to test_predictions.csv file")
    test_result = model.evaluate(test_data)
    if not dummy_test_label:
       accuracy = test_result/len(test_data)
       print("ACC : ", accuracy)
    
    # writing the data into the file 
    file = open('test_predictions.csv', 'w', newline ='') 
    with file:     
        write = csv.writer(file) 
        write.writerows(model.test_predictions) 
    
    print("Evaluating and writing to csv time :", time.time()-eval_write_time)
    
    
    # COMMENT THIS CODE BEFORE SUBMIT.
    if test_data and not dummy_test_label:
        random.shuffle(test_data)
        test_result = model.evaluate(test_data)
        accuracy = test_result/len(test_data)
        
        #print("*" * 80)
        print("Sample of test data            :", len(test_data))
        print("Accuracy on test data is       : ", accuracy)
    
    
    print("Execution time                 : ", time.time() - start_time)
    print("*" * 80)
    #print("Execution time : ", time.time() - start_time)
    


if __name__=="__main__":
    print(len(sys.argv))
    if len(sys.argv) == 4:
        print(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3])
        #main("train_image.csv", "train_label.csv", "test_image.csv")
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main()
        
    