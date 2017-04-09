# loading the data sets (training, validation, test)
# each input set must be: (#images, dim1, dim2, dim3, 1)
# The corresponding label must be: (#images, #classes)
import numpy as np

def load_train_data(image_size, num_channels, label_cnt):

    train_dataset = np.load('Data.npy')
    train_labels = np.load('Label.npy')
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    train_dataset, train_labels = reformat(
            train_dataset, train_labels, image_size, num_channels, label_cnt)
    valid_dataset = train_dataset
    valid_labels = train_labels
#    valid_dataset, valid_labels = reformat(
#            valid_dataset, valid_labels, image_size, num_channels, label_cnt)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    return train_dataset, train_labels, valid_dataset, valid_labels


#def load_test_data(image_size, num_channels, label_cnt):
#    test_dataset = np.arange(32768000).reshape(1000, 32, 32, 32)
#    test_labels = np.concatenate((np.ones(500),2*np.ones(500)))
#    test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_channels, label_cnt)
#    print('Test set', test_dataset.shape, test_labels.shape)
#    return test_dataset, test_labels


def reformat(dataset, labels, image_size, num_channels, label_cnt):
    ''' Reformats the data to the format acceptable for 3D conv layers'''
    dataset = dataset.reshape(
        (-1, image_size, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(label_cnt) == labels[:, None]).astype(np.float32)
    return dataset, labels


def randomize(dataset, labels):
    ''' Randomizes the order of data samples and their corresponding labels'''
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels







