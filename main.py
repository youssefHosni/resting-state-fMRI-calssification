import numpy as np
from hyper_opt import create_mask,model_1D,model_reduced,model_1D_calibrate
import load_data
import data_preprocessing
import generate_result
from sklearn.model_selection import train_test_split

def main():
    # define input file names, directories, and parmaeters
    train_Con_file_name = 'CV_con.npz'
    train_AD_file_name = 'CV_pat.npz'
    #test_Con_file_name = 'CV_ADNI_CON.npz'
    #test_AD_file_name = 'CV_ADNI_AD.npz'
    mask_name = '4mm_brain_mask_bin_epl.nii.gz'
    results_directory = 'Output_results_directory'
    results_path = load_data.find_path(results_directory)
    number_of_cv = 5
    feature_selection_type = 'L2_penality'
    Hyperparameter_model__1 = 1000
    Hyperparameter_model__3 = 1000
    number_of_neighbours = 1
    model_name = 'gaussian_process'

    # loading input data and mask
    train_data,train_labels=load_data.train_data_3d(train_Con_file_name,train_AD_file_name)
    #test_data, test_labels = load_data.test_data_3d(test_Con_file_name, test_AD_file_name)
    mask_4mm = load_data.mask(mask_name)
    original_mask=mask_4mm.get_fdata()


    # data preprocessing
    train_data = np.moveaxis(train_data.copy(), 3, 0)
    #test_data = np.moveaxis(test_data.copy(), 3, 0)
    train_data = train_data * original_mask
    #test_data = test_data * original_mask
    shape = np.shape(train_data)
    train_data_flattened = data_preprocessing.flatten(train_data.copy())
    #test_data_flattened = data_preprocessing.flatten(test_data.copy())
    orignal_mask_flatten = data_preprocessing.flatten(original_mask[np.newaxis, :, :, :].copy())
    orignal_mask_flatten = np.reshape(orignal_mask_flatten, (-1))
    train_data_flattened = data_preprocessing.MinMax_scaler(train_data_flattened.copy())
    #test_data_flattened = data_preprocessing.MinMax_scaler(test_data_flattened.copy())
    # train_data_flattened, test_data_flattened=data_preprocessing.MinMax_scaler_correct(train_data_flattened, test_data_flattened)
    train_data_flattened, test_data_flattened, train_labels, test_labels = train_test_split(train_data_flattened, train_labels, test_size=.2, random_state=42)
    train_data_inlier, train_labels_inlier, outlier_indices_train = data_preprocessing.outliers(train_data_flattened,
                                                                                                train_labels,
                                                                                                number_of_neighbours)
    test_data_inlier, test_labels_inlier, outlier_indices_test = data_preprocessing.novelty(train_data_inlier,
                                                                                            train_labels_inlier,
                                                                                            test_data_flattened,
                                                                                            test_labels,
                                                                                            number_of_neighbours)
    
    train_data_inlier_unflattened = data_preprocessing.deflatten(train_data_inlier, shape)
    train_data_outlier_unflattened = data_preprocessing.deflatten(train_data_flattened[outlier_indices_train], shape)
    train_data_inlier_unflattened = np.moveaxis(train_data_inlier_unflattened.copy(), 0, 3)
    train_data_outlier_unflattened = np.moveaxis(train_data_outlier_unflattened.copy(), 0, 3)
    trian_labels_outliers = train_labels[outlier_indices_train]
    train_data_inlier_noised = data_preprocessing.apply_noise_manytypes(train_data_inlier_unflattened.copy())
    train_data_inlier_filtered = data_preprocessing.apply_filter_manytypes(train_data_inlier_unflattened.copy())
    train_data_inlier_more = data_preprocessing.concatination(train_data_inlier_noised, train_data_inlier_filtered)
    #train_labels_inlier_more = data_preprocessing.dublicate(train_labels_inlier.copy(), 29) #to match length of data
    train_data_outlier_noised = data_preprocessing.apply_noise_manytypes(train_data_outlier_unflattened.copy())
    train_data_outlier_filtered = data_preprocessing.apply_filter_manytypes(train_data_outlier_unflattened.copy())
    train_data_outlier_more = data_preprocessing.concatination(train_data_outlier_noised, train_data_outlier_filtered)
    train_labels_outlier_more = data_preprocessing.dublicate(trian_labels_outliers[:, np.newaxis].copy(), 29)#to match length of data
    train_data_inlier_more = np.moveaxis(train_data_inlier_more.copy(), 3, 0)
    train_data_outlier_more = np.moveaxis(train_data_outlier_more.copy(), 3, 0)
    train_data_outlier_more_flattened = data_preprocessing.flatten(train_data_outlier_more.copy())
    # train_data_inlier_more_flattened = data_preprocessing.flatten(train_data_inlier_more.copy()) #uncomment to use noised inliers
    # train_data_inlier_inlier, train_labels_inlier_inlier, inlier_outlier_indices_train = data_preprocessing.novelty(
    #     train_data_inlier, train_labels_inlier,
    #     train_data_inlier_more_flattened,
    #     train_labels_inlier_more,
    #     number_of_neighbours)
    train_data_outlier_inlier, train_labels_outlier_inlier, outlier_outlier_indices_train = data_preprocessing.novelty(
        train_data_flattened[outlier_indices_train], train_labels[outlier_indices_train],
        train_data_outlier_more_flattened,
        train_labels_outlier_more,
        number_of_neighbours)
    train_data_inlier, train_labels_inlier = data_preprocessing.upsampling(train_data_inlier,train_labels_inlier[:, np.newaxis])


    train_data_inlier, train_labels_inlier = data_preprocessing.shuffling(train_data_inlier,train_labels_inlier)
    train_data_outlier_inlier, train_labels_outlier_inlier = data_preprocessing.upsampling(
                                                                                            train_data_outlier_inlier,
                                                                                            train_labels_outlier_inlier)

    train_data_outlier_inlier, train_labels_outlier_inlier = data_preprocessing.shuffling(train_data_outlier_inlier,
                                                                                            train_labels_outlier_inlier)

    #Brain extraction of data
    train_data_inlier_brain=train_data_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
    test_data_inlier_brain=test_data_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
    train_data_outlier_inlier_brain=train_data_outlier_inlier[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
    test_data_outlier_brain=(test_data_flattened[outlier_indices_test])[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)]
    concated_data = data_preprocessing.concat(train_data_inlier, train_data_outlier_inlier)
    concated_labels = data_preprocessing.concat(train_labels_inlier[:, np.newaxis],
                                                train_labels_outlier_inlier[:, np.newaxis])
    #Model stage 1 with high certainity
    model1_created_mask, model1_, model1_name, model1_weights = create_mask(train_data_inlier_brain, train_labels_inlier,
                                                                            number_of_cv, feature_selection_type,
                                                                            Hyperparameter_model__1, mask_threshold=4,
                                                                            model_type=model_name)
    #train_data_inlier_CVspace = data_preprocessing.coefficient_of_variance(train_data_inlier_brain * model1_created_mask)[:,np.newaxis]
    #test_data_inlier_CVspace = data_preprocessing.coefficient_of_variance(test_data_inlier_brain * model1_created_mask)[:,np.newaxis]
    #train_data_inlier_CVspace = np.sum(train_data_inlier_brain * model1_created_mask, axis=1)[:,np.newaxis]
    #test_data_inlier_CVspace = np.sum(test_data_inlier_brain * model1_created_mask, axis=1)[:,np.newaxis]
    train_data_inlier_CVspace = (train_data_inlier_brain * model1_created_mask)
    test_data_inlier_CVspace = (test_data_inlier_brain * model1_created_mask)

    model1_, model1_name = model_reduced(train_data_inlier_CVspace, train_labels_inlier, model1_created_mask,
                                              data_validation=None, labels_validation=None,
                                              model_type='gaussian_process')
    model1_test_accuracy, model1_F1_score, model1_auc,low_confidence_indices=generate_result.out_result_highprob(test_data_inlier_CVspace,
                                                                                           test_labels_inlier,
                                                                                           original_mask,model1_created_mask,
                                                                                           model1_)
    #Model stage 2 with low certainity
    model2_, model2_name = model_reduced(train_data_inlier_CVspace, train_labels_inlier, model1_created_mask,
                                    data_validation=None, labels_validation=None,
                                    model_type=model_name)
    model2_test_accuracy, model2_F1_score, model2_auc = generate_result.out_result(test_data_inlier_CVspace[low_confidence_indices],
                                                                                             test_labels_inlier[low_confidence_indices],
                                                                                             original_mask,
                                                                                             model1_created_mask,
                                                                                             model2_)
    #Model stage 3 with outliers
    model3_created_mask, model3_, model3_name, model3_weights = create_mask(concated_data,
                                                                            concated_labels, number_of_cv,
                                                                            feature_selection_type, Hyperparameter_model__3,
                                                                            mask_threshold=3,
                                                                            model_type=model_name)
    #concated_data_cv = data_preprocessing.coefficient_of_variance(
     #   concated_data[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)].copy() * model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:, np.newaxis]
    #test_data_outlier_cv = data_preprocessing.coefficient_of_variance(
    #    test_data_outlier_brain *model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])[:, np.newaxis]
    #concated_data_cv = np.sum(
    #                          concated_data[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)].copy() * model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)], axis=1)[:, np.newaxis]
    #test_data_outlier_cv = np.sum(
    #                              test_data_outlier_brain *model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)], axis=1)[:, np.newaxis]
    concated_data_cv = (concated_data[:,np.squeeze(np.where(orignal_mask_flatten>0),axis=0)].copy() * 
                         model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])
    test_data_outlier_cv = (
                                  test_data_outlier_brain *model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)])

    model3_, model3_name = model_reduced(concated_data_cv, concated_labels, model3_created_mask,
                                    data_validation=None, labels_validation=None, model_type=model_name)
    model3_test_accuracy,model3_F1_score,model3_auc = generate_result.out_result(np.nan_to_num(test_data_outlier_cv) ,
                                                                                 np.nan_to_num(test_labels[outlier_indices_test]), np.nan_to_num(original_mask),
                                                                                 np.nan_to_num(model3_created_mask[np.squeeze(np.where(orignal_mask_flatten > 0), axis=0)]), model3_)
    testnum=len(test_labels)
    highcernum=(len(test_labels_inlier)-len(test_labels_inlier[low_confidence_indices]))/testnum
    lowcernum=(len(test_labels_inlier[low_confidence_indices]))/testnum
    outnum=(len(test_labels[outlier_indices_test]))/testnum

    data_preprocessing_method = "Seperating outlier of training set and test set, then synthethise more data from training-outliers, then appling probability predictions. High probability " \
                                "samples model is used with predictions with high probability, then apply low probability model. Finally add noise to outliers and concatinate with inlier data " \
                                "to be used for outlier model"
    generate_result.print_result_3models(mask_4mm, results_path, model3_created_mask[np.squeeze(np.where(orignal_mask_flatten>0),axis=0)],model3_,model3_name,
                                          model3_weights[np.squeeze(np.where(orignal_mask_flatten>0),axis=0)], model3_test_accuracy,
                         model3_auc, model3_F1_score, Hyperparameter_model__3,
                         model2_,model2_name, model2_test_accuracy, model2_auc, model2_F1_score,
                         model1_,model1_created_mask, model1_name, model1_weights, model1_test_accuracy, model1_auc, model1_F1_score,
                         Hyperparameter_model__1,
                         feature_selection_type, data_preprocessing_method,highcernum,lowcernum,outnum)


if __name__=='__main__':
     main()
