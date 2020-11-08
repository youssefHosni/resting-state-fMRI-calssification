import numpy as np
from numpy import load
import data_preprocessing
import preprocessing_methods
train_data_path='/data/fmri/Folder/AD_classification/Data/input_data/Augmented_data/CV_OULU_Con_AD_aug.npz'
train_data = load(train_data_path)['masked_voxels']
test_data_path='/data/fmri/Folder/AD_classification/Data/input_data/CV_ADNI_Con_AD.npz'
test_data = load(test_data_path)['masked_voxels']
train_labels_path='/data/fmri/Folder/AD_classification/Data/input_data/labels/train_labels_aug_data.npz'
train_labels=load(train_labels_path)['labels']
shuffling_indicies = np.random.permutation(len(train_labels))
temp = train_data[:, :, :, shuffling_indicies]
train_data=temp
train_labels = train_labels[shuffling_indicies]

test_labels_path='/data/fmri/Folder/AD_classification/Data/input_data/labels/test_labels.npz'
test_labels=load(test_labels_path)['labels']
shuffling_indicies = np.random.permutation(len(test_labels))
test_data = test_data[:, :, :, shuffling_indicies]
test_labels = test_labels[shuffling_indicies]


#train_data_224=data_preprocessing.size_editing(train_data,224)
#test_data_224=data_preprocessing.size_editing(test_data,224)

train_data_224_preprocessed,test_data_224_preprocessed,train_labels,test_labels=preprocessing_methods.preprocessing(train_data,test_data,train_labels,test_labels,4,0,None,None)
#train_data_224_preprocessed=data_preprocessing.depth_reshapeing(train_data_224_preprocessed)
#test_data_224_preprocessed=data_preprocessing.depth_reshapeing(test_data_224_preprocessed)

print(train_data_224_preprocessed.shape)
print(test_data_224_preprocessed.shape)

transposing_order = [3,0,2,1]
transposing_order = [0,2,1,3]
train_data_ = data_preprocessing.transposnig(train_data, transposing_order)
test_data_ = data_preprocessing.transposnig(test_data, transposing_order)

np.savez('/data/fmri/Folder/AD_classification/Data/input_data/preprocessed_data/CV_OULU_Con_AD_preprocessed_224_3.npz',masked_voxels=train_data_224_preprocessed)
np.savez('/data/fmri/Folder/AD_classification/Data/input_data/preprocessed_data/CV_ADNI_Con_AD_preprocessed_224_3.npz',masked_voxels=test_data_224_preprocessed)
