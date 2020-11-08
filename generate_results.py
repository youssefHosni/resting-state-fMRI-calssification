import numpy as np
import os
import nibabel as nib
from datetime import date
from sklearn import metrics
from sklearn.metrics import f1_score
import pickle
from sklearn.metrics import confusion_matrix
def out_result(test_data,test_labels,original_mask,created_mask,model):
    if (np.shape(test_data)[1]==1):
         test_data=np.reshape(test_data,(-1,1))
         predicted_labels = model.predict(test_data)
         test_accuracy = model.score(test_data,test_labels[:,np.newaxis])
    else:
         print('test_data',np.shape(test_data))
         print('created_mask',np.shape(created_mask))
         masked_test_data = test_data*created_mask
         predicted_labels = model.predict(masked_test_data)
         test_accuracy = model.score(masked_test_data,
                                        test_labels[:, np.newaxis])
    F1_score = f1_score(test_labels,predicted_labels, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
    auc=metrics.auc(fpr, tpr)
    if ((auc<.5)& (~np.isnan(auc))):
        auc = .5+(.5-auc)
    if ((test_accuracy==0) & np.isnan(auc)):
        auc = 0.0
    if np.isnan(auc):
        auc = 1.0

    cm_test = confusion_matrix(test_labels, predicted_labels)
    print(cm_test)
    return test_accuracy, F1_score, auc
def out_result_highprob(test_data,test_labels,original_mask,created_mask,model):
    upper_bound=.64
    lower_bound=.35
    if (np.shape(test_data)[1]==1):
         test_data=np.reshape(test_data,(-1,1))
         predicted_labels = model.predict(test_data)
         predicted_prob=model.predict_proba(test_data)
         if (np.shape(test_data[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))])[0]==0):
             return 0,0,0,range(len(predicted_labels))
         test_accuracy = model.score(test_data[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))],(test_labels[:,np.newaxis])[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))])
    else:
         print('test_data',np.shape(test_data))
         print('created_mask',np.shape(created_mask))
         masked_test_data = test_data*created_mask
         predicted_labels = model.predict(masked_test_data)
         predicted_prob = model.predict_proba(masked_test_data)
         if (np.shape((masked_test_data)[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))])[0]==0):
             return 0,0,0,range(len(predicted_labels))
         test_accuracy = model.score((masked_test_data)[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))],(test_labels[:,np.newaxis])[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))])
    F1_score = f1_score(test_labels[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))],predicted_labels[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))], average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(test_labels[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))], predicted_labels[np.where((predicted_prob[:,1]>upper_bound) | (predicted_prob[:,1]<lower_bound))])
    auc=metrics.auc(fpr, tpr)
    low_confidence_indices = np.where((predicted_prob[:, 1] < upper_bound) & (predicted_prob[:, 1] > lower_bound))
    if ((auc < .5) & (~np.isnan(auc))): auc = .5 + (.5 - auc)
    if ((test_accuracy == 0) & np.isnan(auc)): auc = 0.0
    if np.isnan(auc): auc = 1.0
    if (np.shape(list(low_confidence_indices))[1]==0): low_confidence_indices=0
    return test_accuracy,F1_score,auc,low_confidence_indices
def print_result_3models(original_mask,results_directory,created_mask_3,model_3,model_name_3,weights_3,model3_accuracy,model3_auc,model3_f1,Hyperparameter_3,
                         model_2,model_name_2,model2_accuracy,model2_auc,model2_f1,
                         model_1,created_mask_1,model_name_1,weights_1,model1_accuracy,model1_auc,model1_f1,Hyperparameter_1,
                         feature_selection_type,data_preprocessing_method,highcernum,lowcernum,outliernum):

    today = str(date.today()) # To save the results in a directory with the date as a name

    if os.path.exists(os.path.join(results_directory,today))==0:
        os.mkdir(os.path.join(results_directory,today))

    if len(os.listdir(os.path.join(results_directory,today))) ==0:
        file_number = 1
    else:

        #latest_file = sorted(os.path.join(results_directory,today),key=x,reverse=True)
        print(os.path.join(results_directory, today))
        dir_list=os.listdir(os.path.join(results_directory,today))
        latest_file=sorted(list(map(int,dir_list)),reverse=True)
        print(latest_file)

        file_number = ((latest_file[0]))+1
    div=3
    total=highcernum+lowcernum+outliernum
    print('total,highcernum,lowcernum,outliernum ',total,highcernum,lowcernum,outliernum)
    #if highcernum==0:
    #    div-=1
    #if outliernum==0:
    #    div-=1
    highcernum=highcernum/total
    lowcernum=lowcernum/total
    outliernum=outliernum/total
    os.mkdir(os.path.join(results_directory,today,str(file_number)))
    line1 = 'Test accuracy of high certainty model :' + '  ' + str(model1_accuracy)
    line2 = 'F1 score of the high certainty model' + '  ' + str(model1_f1)
    line3 = 'AUC of high certainty model :' + '  ' + str(model1_auc)
    line4 = 'Test accuracy of low certainty model:' + '  ' + str(model2_accuracy)
    line5 = 'F1 score of low certainty model:' + '  ' + str(model2_f1)
    line6 = 'AUC of low certainty model :' + '  ' + str(model2_auc)
    line7 = 'Test accuracy outliers model:' + '  ' + str(model3_accuracy)
    line8 = 'F1 score of outliers model:' + '  ' + str(model3_f1)
    line9 = 'AUC of outliers model :' + '  ' + str(model3_auc)
    line10 = 'Test accuracy total:' + '  ' + str((highcernum*model1_accuracy+lowcernum*model2_accuracy+outliernum*model3_accuracy))
    line11 = 'F1 score of total' + '  ' + str((highcernum*model1_f1+lowcernum*model2_f1+outliernum*model3_f1))
    line12 = 'AUC of total :' + '  ' + str((highcernum*model1_auc+lowcernum*model2_auc+outliernum*model3_auc))
    f = open(os.path.join(results_directory,today,str(file_number),'Results.txt'),"w+")
    f.write("{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" .format(
                                                                                line1, line2, line3,line4,line5,line6,
                                                                                line7,line8,line9,line10,line11,line12))


    line1='The model used to obtain high certainty model result is '+ '  '+ model_name_1
    line2 = 'the Max_num_of_features used in high certainty model is ' + '  ' + str(Hyperparameter_1)
    line3='The model used to obtain low certainty model result is '+ '  '+ model_name_2
    line4='The model used to obtain outliers model result is '+ '  '+ model_name_3
    line5 = 'the Max_num_of_features used in outliers model is ' + '  ' + str(Hyperparameter_3)
    line6='The feature selection methods is ' + '  '+ feature_selection_type
    line7= 'The preprocessing method used is '+ ' '+ data_preprocessing_method
    f=open(os.path.join(results_directory,today,str(file_number),'README.txt'),"w+")
    f.write("{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n" "{}" "\n"  .format(line1,line2,line3,line4,line5,line6,line7))

    mask_print(original_mask,created_mask_1,os.path.join(results_directory,today,str(file_number)),'high_certainity_model_')
    weights_print(original_mask ,weights_1, os.path.join(results_directory,today,str(file_number)),'high_certainity_model_')
    mask_print(original_mask, created_mask_3, os.path.join(results_directory, today, str(file_number)),'outliers_model_')
    weights_print(original_mask, weights_3, os.path.join(results_directory, today, str(file_number)),'outliers_model_')
    filename = 'high_certainity_model.sav'
    pickle.dump(model_1, open(os.path.join(os.path.join(results_directory, today, str(file_number)),filename), 'wb'))
    filename = 'low_certainty_model.sav'
    pickle.dump(model_2, open(os.path.join(os.path.join(results_directory, today, str(file_number)),filename), 'wb'))
    filename = 'outliers_model.sav'
    pickle.dump(model_3, open(os.path.join(os.path.join(results_directory, today, str(file_number)),filename), 'wb'))
    return


def mask_print(original_mask,created_mask,output_dir,name):

    masking_shape = original_mask.shape
    masking = np.empty(masking_shape, dtype=float)
    masking[:, :, :] = original_mask.get_data().astype(float)
    masking[np.where(masking > 0)] = masking[np.where(masking > 0)] * 0 + created_mask
    hdr = original_mask.header
    aff = original_mask.affine
    out_img = nib.Nifti1Image(masking, aff, hdr)
    nib.save(out_img, os.path.join(output_dir,name+'mask.nii.gz'))
    return

def weights_print(original_mask, weights, output_dir,name):

    masking_shape = original_mask.shape
    masking = np.empty(masking_shape, dtype=float)
    masking[:, :, :] = original_mask.get_data().astype(float)
    masking[np.where(masking > 0)] = masking[np.where(masking > 0)] * 0 + weights
    hdr = original_mask.header
    aff = original_mask.affine
    out_img = nib.Nifti1Image(masking, aff, hdr)
    nib.save(out_img, os.path.join(output_dir, name+'weights.nii.gz'))
    return

def confidence_interval_element_95(element,name):
    alpha = 5.0
    lower_p = alpha / 2.0
    upper_p = (100 - alpha) + (alpha / 2.0)
    print('for'+name)
    print('50th percentile (median) = %.3f' % np.median(element))
    lower = max(0.0, np.percentile(element, lower_p))
    print('%.1fth percentile = %.3f' % (lower_p, lower))
    upper = min(1.0, np.percentile(element, upper_p))
    print('%.1fth percentile = %.3f' % (upper_p, upper))
    print()
    return
def confidence_interval_model_95(accuracy,f1_score,auc,name):
    confidence_interval_element_95(accuracy, name+' accuracy ')
    confidence_interval_element_95(f1_score, name+' F1_score ')
    confidence_interval_element_95(auc, name+' AUC ')
    return
def confidence_interval_element_99(element,name):
    alpha = 1.0
    lower_p = alpha / 2.0
    upper_p = (100 - alpha) + (alpha / 2.0)
    print('for'+name)
    print('50th percentile (median) = %.3f' % np.median(element))
    lower = max(0.0, np.percentile(element, lower_p))
    print('%.1fth percentile = %.3f' % (lower_p, lower))
    upper = min(1.0, np.percentile(element, upper_p))
    print('%.1fth percentile = %.3f' % (upper_p, upper))
    print()
    return
def confidence_interval_model_99(accuracy,f1_score,auc,name):
    confidence_interval_element_99(accuracy, name+' accuracy ')
    confidence_interval_element_99(f1_score, name+' F1_score ')
    confidence_interval_element_99(auc, name+' AUC ')
    return
