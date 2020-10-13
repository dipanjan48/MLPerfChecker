import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class preprocessing:

    def __init__(self,df1,dfsub,var_dependent):
        self.df1 = df1           
        self.dfsub = dfsub #submission file
        self.var_dependent = var_dependent
    
    #def file_import(self):
        #df = pd.read_csv(self.file_path)
        #return df
    
    def missing_values(self):
        df = self.df1
        df_sub = self.dfsub
        del_col = []
        for i in df.columns:
            if int(df[i].isnull().sum()) != 0:
                if float(df[i].isnull().sum()) > 0.4 * len(df.index):
                    df.drop([i] , axis = 1 , inplace = True)
                    del_col.append(i)
                else:
                    df[i] = df[i].fillna(df[i].mode()[0])
        
        for col in df.columns:
            if df[col].nunique()>0.7 * len(df.index):
                df.drop([col] , axis = 1 , inplace = True)
                del_col.append(col)
        
        for w in [del_col]:
            df_sub.drop([w] , axis = 1 , inplace = True)
        for q in df_sub.columns:
            df[q] = df[q].fillna(df[q].mode()[0])
        return (df,df_sub)
    
    #encoding independent variable
    def encode_independent_var(self):
        df,df_sub = self.missing_values()
        final_df = df.copy()
        final_df.drop([self.var_dependent] , axis=1 , inplace=True)   #independent variable
        #Y = df[var_dependent]                                   #dependent variable
        multcolumns = set(final_df.columns) - set(final_df._get_numeric_data().columns)
        #print (multcolumns)
        df_final=final_df
        i=0
        for fields in multcolumns:
            
            #print(fields)
            df1=pd.get_dummies(final_df[fields],drop_first=True)
            
            final_df.drop([fields],axis=1,inplace=True)
            if i==0:
                df_final=df1.copy()
            else:
                
                df_final=pd.concat([df_final,df1],axis=1)
            i=i+1            
        df_final=pd.concat([final_df,df_final],axis=1)

        #encoding submission set
        final_dfsub = df_sub.copy()
        #final_df.drop([self.var_dependent] , axis=1 , inplace=True)   #independent variable
        #Y = df[var_dependent]                                   #dependent variable
        multcolumns_sub = set(final_dfsub.columns) - set(final_dfsub._get_numeric_data().columns)
        #print (multcolumns)
        df_finalsub=final_dfsub
        i=0
        for fields in multcolumns_sub:
            
            #print(fields)
            df1=pd.get_dummies(final_dfsub[fields],drop_first=True)
            
            final_dfsub.drop([fields],axis=1,inplace=True)
            if i==0:
                df_finalsub=df1.copy()
            else:
                
                df_finalsub=pd.concat([df_finalsub,df1],axis=1)
            i=i+1            
        df_finalsub=pd.concat([final_dfsub,df_finalsub],axis=1)
        return (df_final, df_finalsub)
    
    #encoding dependent variable
    def encode_dependent_var(self):
        df = self.missing_values()
        Y = df[self.var_dependent]
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        return (Y)
    
    #splitting dataset
    def split(self):
        X = self.encode_independent_var()
        Y = self.encode_dependent_var()
        X_train, X_test , Y_train , Y_test = train_test_split(X,Y ,test_size = 0.2)
        return (X_train, X_test , Y_train , Y_test)




