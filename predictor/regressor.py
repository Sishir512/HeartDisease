
# Importing libraries 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import warnings 
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings( "ignore" ) 
  
# to compare our model's accuracy with sklearn model 
from sklearn.linear_model import LogisticRegression 
# Logistic Regression 
class LogitRegression(): 
    def __init__( self, learning_rate, iterations ) :         
        self.learning_rate = learning_rate         
        self.iterations = iterations 
          
    # Function for model training     
    def fit( self, X, Y ) :         
        # no_of_training_examples, no_of_features         
        self.m, self.n = X.shape         
        # weight initialization         
        self.W = np.zeros( self.n )         
        self.b = 0        
        self.X = X         
        self.Y = Y 
          
        # gradient descent learning 
                  
        for i in range( self.iterations ) :             
            self.update_weights()             
        return self
      
    # Helper function to update weights in gradient descent 
      
    def update_weights( self ) :            
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) ) 
          
        # calculate gradients         
        tmp = ( A - self.Y.T )         
        tmp = np.reshape( tmp, self.m )         
        dW = np.dot( self.X.T, tmp ) / self.m          
        db = np.sum( tmp ) / self.m  
          
        # update weights     
        self.W = self.W - self.learning_rate * dW     
        self.b = self.b - self.learning_rate * db 
          
        return self
      
    # Hypothetical function  h( x )  
      
    def predict( self, X ) :     
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )         
        Y = np.where( Z > 0.5, 1, 0 )         
        return Y , Z
  
  
# Driver code 
  
def main() : 
      
    # Importing dataset     
    df = pd.read_csv( "C:\\Users\\Sishir512\\Desktop\\Heart Disease\\heartdisease\\predictor\\heartdata.csv" ) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1:].values 
    
    #scaler = MinMaxScaler(feature_range=(0, 1)) 
    #X=scaler.fit_transform(X)
    print(' x  is' , len(X))
    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split( 
      X, Y, test_size = 1/3, random_state = 0 ) 
      
    # Model training     
    model = LogitRegression( learning_rate = 0.0001, iterations = 1000 ) 
      
    model.fit( X_train, Y_train )     
    model1 = LogisticRegression()     
    model1.fit( X_train, Y_train) 
      
    # Prediction on test set 
    
    Y_pred , Y_x = model.predict(X_test)
    c1=0
    c2=0
    f=0
    
    for i in range(len(Y_pred)):
        if Y_pred[i]==Y_test[i]:
            c1=c1+1
            
        else:
            f=f+1
        
    print("Total correct predictions ",c1)
    '''
    fuck = model1.predict(X_test)
    x=0
    

    for i in range(len(fuck)):
        if fuck[i]==Y_test[i]:
            x=x+1
    print((c1/len(Y_pred)*100))
    print((x/len(Y_pred)*100))
    '''
    tp=0
    fp=0
    fn=0
    
    for i in range(len(Y_pred)):
        if Y_test[i]==1 and Y_pred[i]==1:
            tp+=1
        if Y_test[i]==0 and Y_pred[i]==1:
            fp+=1
        if Y_test[i]==1 and Y_pred[i]==0:
            fn+=1
    
    precision = tp/(tp+fp)
    recall=tp/(tp+fn)
    f_score=(2 * precision * recall)/(precision + recall)
    print("True Positive ",tp)
    print("False Positive ",fp)
    print("False Negative ",fn)
    print("Precision ",precision)
    print("Recall ",recall)
    print("F-score ",f_score)
    print("Total test data ", len(Y_test))
    print("Total train data ", len(Y_train))
    '''print(len(Y_test))
    for i in range(len(Y_test)):
        print(Y_test[i] , " : " , Y_pred[i])
    '''
    print('-----------------')
    
    '''Y_pred , Y_x= model.predict(np.array([48,1,2,124,255,1,1,175,0,0,2,2,2]))
    z= model1.predict_proba(np.array([48,1,2,124,255,1,1,175,0,0,2,2,2]).reshape(1,-1))'''

    '''
    Y_pred , Y_x= model.predict(np.array([53,0,0,130,264,0,0,143,0,0.4,1,0,2]))
    z= model1.predict_proba(np.array([67,1,2,152,212,0,0,150,0,0.8,1,0,3]).reshape(1,-1))
    #print(z)
    print(Y_x)
    '''
    x , y = model.predict(np.array([37 , 1 , 2 ,130 , 250 , 0 , 1 , 187 , 0 , 3.5 , 0 , 0 , 2]))
    print("fINALLY " , x ," ",  y)
    

def find():
    df = pd.read_csv( "C:\\Users\\Sishir512\\Desktop\\Heart Disease\\heartdisease\\predictor\\heartdata.csv" ) 
    X = df.iloc[:,:-1].values 
    Y = df.iloc[:,-1:].values
    return X,Y
      
    
  
if __name__ == "__main__" :      
    main()
