import argparse
import random,os,sys
import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras import optimizers,utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History
from keras.utils import multi_gpu_model
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
import hickle as hkl
import scipy.sparse as sp
import argparse
from model import KerasMultiSourceDualGCNModel

unit_list = [256] 
israndom = False 
Drug_info_file = '../data/drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = '../data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = '../data/drug/drug_graph_feat'
Cancer_response_exp_file = '../data/CCLE/GDSC_IC50.csv'
PPI_file = "../data/PPI/PPI_network.txt"
selected_info_common_cell_lines = "../data/CCLE/cellline_list.txt"
selected_info_common_genes = "../data/CCLE/gene_list.txt"
celline_feature_folder = "../data/CCLE/omics_data"

Max_atoms = 100
TCGA_label_set = ["ALL","BLCA","BRCA","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ','SARC','UCEC','MESO', 'PRAD']

def MetadataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,PPI_file,selected_info_common_cell_lines,selected_info_common_genes):
    with open(selected_info_common_cell_lines) as f:
        common_cell_lines = [item.strip() for item in f.readlines()]
    
    with open(selected_info_common_genes) as f:
        common_genes = [item.strip() for item in f.readlines()]
    idx_dic={}
    for index, item in enumerate(common_genes):
        idx_dic[item] = index

    ppi_adj_info = [[] for item in common_genes] 
    for line in open(PPI_file).readlines():
        gene1,gene2 = line.split('\t')[0],line.split('\t')[1]
        if idx_dic[gene1]<=idx_dic[gene2]:
            ppi_adj_info[idx_dic[gene1]].append(idx_dic[gene2])
            ppi_adj_info[idx_dic[gene2]].append(idx_dic[gene1])

    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}

    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label

    drug_pubchem_id_set = []
    drug_feature = {} 
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())

    IC50_df = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    drug_match_list=[item for item in IC50_df.index if item.split(':')[1] in drugid2pubchemid.keys()]
    IC50_df = IC50_df.loc[drug_match_list]
    
    index_name = [drugid2pubchemid[item.split(':')[1]] for item in IC50_df.index if item.split(':')[1] in drugid2pubchemid.keys()]
    IC50_df.index = index_name
    redundant_names = list(set([item for item in IC50_df.index if list(IC50_df.index).count(item)>1]))
    retain_idx = []
    for i in range(len(IC50_df.index)):
        if IC50_df.index[i] not in redundant_names:
            retain_idx.append(i)
    IC50_df = IC50_df.iloc[retain_idx]

    data_idx = [] 
    for each_drug in IC50_df.index:
        for each_cellline in IC50_df.columns:
            if str(each_drug) in drug_pubchem_id_set and each_cellline in common_cell_lines:
                if not np.isnan(IC50_df.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys() and cellline2cancertype[each_cellline] in TCGA_label_set:
                    ln_IC50 = float(IC50_df.loc[each_drug,each_cellline])
                    data_idx.append((each_cellline,each_drug,ln_IC50,cellline2cancertype[each_cellline]))
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))
    return ppi_adj_info, drug_feature, data_idx 

def DataSplit(data_idx,TCGA_label_set,n_splits=5):
    # n_split: number of CV
    data_train_idx,data_test_idx = [[] for i in range(n_splits)] , [[] for i in range(n_splits)]
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        idx = 0
        for train, test in kf.split(data_subtype_idx):
            data_train_idx[idx] += [data_subtype_idx[item] for item in train]
            data_test_idx[idx] += [data_subtype_idx[item] for item in test]
            idx += 1
    return data_train_idx,data_test_idx

# Normalize adjacent matrix D^{-0.5}{T}A^{T}D^{-0.5}
def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix

def CalculateGraphFeat(feat_mat,adj_list,israndom=False):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0]) 
    feat[:feat_mat.shape[0],:] = feat_mat  
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1 
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2
    return [feat,adj_mat]

def CelllineGraphAdjNorm(ppi_adj_info,selected_info_common_genes):
    with open(selected_info_common_genes) as f:
        common_genes = [item.strip() for item in f.readlines()]
    nb_nodes = len(common_genes)
    adj_mat = np.zeros((nb_nodes,nb_nodes),dtype='float32')
    for i in range(len(ppi_adj_info)):
        nodes = ppi_adj_info[i]
        for each in nodes:
            adj_mat[i,each] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    norm_adj = NormalizeAdj(adj_mat)
    return norm_adj 

def FeatureExtract(data_idx,drug_feature, celline_feature_folder, selected_info_common_cell_lines, selected_info_common_genes,israndom=False):
    cancer_type_list = []
    nb_instance = len(data_idx)
    drug_data = [[] for item in range(nb_instance)]
    cell_line_data_feature = [[] for item in range(nb_instance)]
    target = np.zeros(nb_instance,dtype='float32')
    cellline_drug_pair = []
    with open(selected_info_common_cell_lines) as f:
        common_cell_lines = [item.strip() for item in f.readlines()]
    
    with open(selected_info_common_genes) as f:
        common_genes = [item.strip() for item in f.readlines()]
    dic_cell_line_feat = {}
    for each in common_cell_lines:
        dic_cell_line_feat[each] = pd.read_csv('%s/%s.csv'%(celline_feature_folder, each), index_col=0).loc[common_genes].values 
    for idx in range(nb_instance):
        cell_line_id,pubchem_id,ln_IC50,cancer_type = data_idx[idx]
        cellline_drug_tmp = cell_line_id + "_" + pubchem_id
        cellline_drug_pair.append(cellline_drug_tmp)
        cell_line_feat_mat =  dic_cell_line_feat[cell_line_id] 
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)] 
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list,israndom)
        cell_line_data_feature[idx] = cell_line_feat_mat
        target[idx] = ln_IC50
        cancer_type_list.append(cancer_type)
    drug_feat = np.array([item[0] for item in drug_data])
    drug_adj = np.array([item[1] for item in drug_data])
    return drug_feat,drug_adj, np.array(cell_line_data_feature),target,cancer_type_list

class MyCallback(Callback):
    def __init__(self,validation_data,result_file_path,patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
        self.result_file_path = result_file_path
    def on_train_begin(self,logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save(self.result_file_path)
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        pcc_val = pearsonr(self.y_val, y_pred_val[:,0])[0]
        print('pcc-val: %s' % str(round(pcc_val,4)))
        if pcc_val > self.best:
            self.best = pcc_val
            self.wait = 0
            self.best_weight = self.model.get_weights()
        else:
            self.wait+=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        return

def ModelTraining(model,X_train,Y_train,validation_data,result_file_path,result_file_path_callback, batch_size=32,nb_epoch=100):
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = optimizer,loss='mean_squared_error',metrics=['mse'])
    callbacks = [ModelCheckpoint(result_file_path,monitor='val_loss',save_best_only=True, save_weights_only=False), MyCallback(validation_data=validation_data,result_file_path=result_file_path_callback,patience=20)]
    model.fit(x=X_train,y=Y_train,batch_size=batch_size,epochs=nb_epoch, validation_data=validation_data,callbacks=callbacks)
    return model

def ModelEvaluate(model,X_val,Y_val,cancer_type_test_list,data_test_idx_current,file_path_pcc_log,file_path_spearman_log,file_path_rmse_log, file_path_csv,batch_size=32):
    Y_pred = model.predict(X_val,batch_size=batch_size)
    overall_pcc = pearsonr(Y_pred[:,0],Y_val)[0]
    overall_rmse = mean_squared_error(Y_val,Y_pred[:,0],squared=False)
    overall_spearman = spearmanr(Y_pred[:,0],Y_val)[0]
    f_out_pcc = open(file_path_pcc_log,'w')
    f_out_rmse = open(file_path_spearman_log,'w')
    f_out_spearman = open(file_path_rmse_log,'w')
    cancertype2pcc = {}
    cancertype2rmse = {}
    cancertype2spearman = {}
    for each in TCGA_label_set:
        ind = [b for a,b in zip(cancer_type_test_list,list(range(len(Y_pred)))) if a==each]
        if len(ind)>1:
            cancertype2pcc[each] = pearsonr(Y_pred[:,0][ind],Y_val[ind])[0]
            f_out_pcc.write('%s\t%d\t%.4f\n'%(each,len(ind),cancertype2pcc[each]))

            cancertype2rmse[each] = mean_squared_error(Y_pred[:,0][ind],Y_val[ind],squared=False)
            f_out_rmse.write('%s\t%d\t%.4f\n'%(each,len(ind),cancertype2rmse[each]))

            cancertype2spearman[each] = spearmanr(Y_pred[:,0][ind],Y_val[ind])[0]
            f_out_spearman.write('%s\t%d\t%.4f\n'%(each,len(ind),cancertype2spearman[each]))

    f_out_pcc.write("AvegePCC\t%.4f\n"%overall_pcc)
    f_out_rmse.write("AvegeRMSE\t%.4f\n"%overall_rmse)
    f_out_spearman.write("AvegeSpearman\t%.4f\n"%overall_spearman)
    f_out_pcc.close()
    f_out_rmse.close()
    f_out_spearman.close()

    f_out = open(file_path_csv,'w')
    f_out.write('drug_id,cellline_id,cancer_type,IC50,IC50_predicted\n')
    for i in range(len(cancer_type_test_list)):
        drug_ = data_test_idx_current[i][1]
        cellline_ = data_test_idx_current[i][0]
        predicted_ = Y_pred[i,0]
        true_ = Y_val[i]
        cancertype_ = cancer_type_test_list[i]
        f_out.write('%s,%s,%s,%.4f,%.4f\n'%(drug_,cellline_,cancertype_,true_,predicted_))
    f_out.close()
    return cancertype2pcc

def main():
    np.random.seed(123)
    random.seed(123)
    batch_size = 128
    nb_epoch = 500
    ppi_adj_info, drug_feature, data_idx = MetadataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,PPI_file,selected_info_common_cell_lines, selected_info_common_genes)
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info,selected_info_common_genes)
    data_train_idx, data_test_idx = DataSplit(data_idx,TCGA_label_set, n_splits=1)
    data_train_idx, data_test_idx = data_train_idx[0], data_test_idx[0]

    X_train_drug_feat,X_train_drug_adj,X_train_cellline_feat,Y_train,cancer_type_train_list=FeatureExtract(data_train_idx,drug_feature,celline_feature_folder,selected_info_common_cell_lines, selected_info_common_genes,israndom)
    X_train_cellline_feat_mean = np.mean(X_train_cellline_feat, axis=0)
    X_train_cellline_feat_std = np.std(X_train_cellline_feat, axis=0)
    X_train_cellline_feat = (X_train_cellline_feat - X_train_cellline_feat_mean) / X_train_cellline_feat_std
    X_train = [X_train_drug_feat,X_train_drug_adj,X_train_cellline_feat,np.array([ppi_adj for i in range(X_train_drug_feat.shape[0])])]

    X_test_drug_feat,X_test_drug_adj,X_test_cellline_feat,Y_test,cancer_type_test_list=FeatureExtract(data_test_idx,drug_feature,celline_feature_folder,selected_info_common_cell_lines, selected_info_common_genes,israndom)
    X_test_cellline_feat = (X_test_cellline_feat - X_train_cellline_feat_mean) / X_train_cellline_feat_std
    X_test = [X_test_drug_feat,X_test_drug_adj,X_test_cellline_feat,np.array([ppi_adj for i in range(X_test_drug_feat.shape[0])])]

    test_data = [X_test, Y_test]

    model = KerasMultiSourceDualGCNModel().createMaster(X_train[0][0].shape[-1],X_train[2][0].shape[-1],unit_list)

    print("... Train the model ...")
    model = ModelTraining(model=model,
                          X_train=X_train,
                          Y_train=Y_train,
                          validation_data=test_data,
                          result_file_path='../checkpoint/best_DualGCNmodel.h5',
                          result_file_path_callback='../checkpoint/MyBestDualGCNModel.h5',
                          batch_size=batch_size,
                          nb_epoch=nb_epoch)
    print("... Evaluate the model ...")
    cancertype2pcc = ModelEvaluate(model=model,
                                  X_val=X_test,
                                  Y_val=Y_test,
                                  cancer_type_test_list=cancer_type_test_list,
                                  data_test_idx_current=data_test_idx,
                                  file_path_pcc_log='../log/pcc_DualGCNmodel.log',
                                  file_path_spearman_log='../log/spearman_DualGCNmodel.log',
                                  file_path_rmse_log='../log/rmse_DualGCNmodel.log',
                                  file_path_csv='../log/result_DualGCNmodel.csv',
                                  batch_size=batch_size)
    print('Evaluation finished!')

if __name__ == "__main__":
    main()
