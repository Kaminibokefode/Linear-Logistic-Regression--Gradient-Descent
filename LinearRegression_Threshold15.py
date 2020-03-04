import numpy as np
import matplotlib.pyplot as plt

#step1
def initialize_parameters(lenw):
    w=np.random.randn(1,lenw)
    #w=np.zeroes((1,lenw))
    b=0
    return w,b

#step2
def forward_prop(X,w,b): #w-->1xn , X-->nxm
    z=np.dot(w,X)+b #z--> 1xm b_vector = [b b b b .....]
    return z


#step 3
def cost_function(z,y):
    m=y.shape[1]
    J=(1/(2*m))*np.sum(np.square(z-y))
    return J

#step4
def back_prop(X,y,z):
    m=y.shape[1]
    dz = (1/m)*(z-y)
    dw = np.dot(dz,X.T) #dw --> 1xn
    db = np.sum(dz)
    return dw,db
    
    
#step5
def gradient_descent_update(w,b,dw,db,learning_rate):
    w=w-learning_rate*dw
    b=b-learning_rate*db
    return w,b

#step6
def linear_regression_model(X_train,y_train,X_val,y_val,learning_rate,epochs,threshold):
    
    lenw = X_train.shape[0]
    w,b= initialize_parameters(lenw) #step1
    th=threshold
    iteration=2
    
    costs_train = []
    costs_val   = []
    m_train     = y_train.shape[1]
    m_val       = y_val.shape[1]
    
    for i in range(1,epochs+1):
        z_train    = forward_prop(X_train,w,b) #step2
        cost_train = cost_function(z_train,y_train) #step3
        dw,db      = back_prop(X_train,y_train,z_train) #step4
        w,b        = gradient_descent_update(w,b,dw,db,learning_rate) #step5
        
        count=i
        #store trining cost in a list for plotting purpose
        #if i%10==0:
        costs_train.append(cost_train)
            
        #MAE_train
        MAE_train = (1/m_train)*np.sum(np.abs(z_train-y_train))
        
        #cost_val,MAE val
        z_val    = forward_prop(X_val,w,b)
        cost_val = cost_function(z_val,y_val)
        MAE_val  = (1/m_val)*np.sum(np.abs(z_val-y_val))
        #if i%10==0:
        costs_val.append(cost_val)
        
        
        
        if count>1:
            diff = (costs_train[-2]-costs_train[-1])
            iteration=iteration+1
            if diff <th:                
                break
            else:
                continue

        

        #print out cost_train,cost_val,MAE_train,MAE_val
        
        #print('Epochs '+str(i)+'/'+str(epochs)+': ')
        #print('Training cost '+str(cost_train)+'|'+'Validation cost '+str(cost_val))
        #print('Training MAE '+str(MAE_train)+'|'+'Validation MAE '+str(MAE_val))
        #print('Coefficients '+str(w))
        
        
    
    #plt.plot(costs_train,label='tc_lr='+str(learning_rate))
    #plt.plot(costs_val,label='vc_lr='+str(learning_rate))
    #plt.xlabel('Iterations(per tens)')
    #plt.ylabel('Cost')
    #plt.title('Learning rate : ')
    #plt.legend()
    #plt.show

   
    #print('Coefficients '+str(w))
    
   
    return costs_train,costs_val,w,iteration
        