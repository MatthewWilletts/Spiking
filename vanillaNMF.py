import numpy as np

#Create random array

RandomTestData = np.random.rand(5,10)


def vanilla_nmf(V_data,r,number_of_iterations):

    #Function to calculate the standard Non-negative matrix factorisation on a matrix of data, as per Lee and Seung 2001

    #Extract size of out Vdata array
    V_data_size = V_data.shape

    #Create our W matrix - this will encode our basis vectors
    W_data = np.random.rand(V_data_size[0],r)
    W_data_temp = W_data

    #Create our H matrix - this says where and how to combine our basis vectors
    H_data = np.random.rand(r,V_data_size[1])
    H_data_temp = H_data

    # #Now we run the update rule for number_of_iterations
    # W_row_sum = np.sum(W_data, axis=1)
    # W_col_sum = np.sum(W_data, axis=0)
    #
    # H_row_sum = np.sum(H_data, axis=1)
    # H_col_sum = np.sum(H_data, axis=0)

    k=0
    while k<number_of_iterations:

        for i in range(0,V_data_size[0]):
            for a in range(0,r):
                for u in range(0,V_data_size[1]):
                    WH_data_i = np.dot(W_data, H_data)[i,]
                    WH_data_u = np.dot(W_data, H_data)[..., u]


                    HV_elementwise_product_vector = np.multiply(H_data[a,], V_data[i,])
                    WV_elementwise_product_vector = np.multiply(W_data[..., a], V_data[..., u])

                    W_data_temp[i,a]=W_data[i,a]*np.sum(np.divide(HV_elementwise_product_vector,WH_data_i))/np.sum(H_data[a,])

                    H_data_temp[a,u]=H_data[a,u]*np.sum(np.divide(WV_elementwise_product_vector,WH_data_u))/np.sum(W_data[..., a])
        k = k+1
        W_data = W_data_temp
        H_data = H_data_temp

    return W_data, H_data, k;


def shift_operator(array,n):
    e = np.zeros_like(array)
    if n >0:
        e[..., n:] = array[..., :-n]
    elif n == 0:
        e = array
    else:
        e[..., :n] = array[..., -n:]
    return e

def calculate_lambda(WT_Data,H_data,T):

    value_of_lambda = np.zeros_like(np.dot(WT_Data[..., ..., 1], H_data))

    for t in range(0, T):
        value_of_lambda=value_of_lambda+np.dot(WT_Data[..., ..., t], shift_operator(H_data,t))

    return value_of_lambda

def conv_nmf(V_data,r,T,number_of_iterations):

    #Function to calculate the standard Non-negative matrix factorisation on a matrix of data, as per Lee and Seung 2001

    #Extract size of out Vdata array
    V_data_size = V_data.shape

    #Create our W matrix - this will encode our basis vectors
    W_data = np.random.rand(V_data_size[0],r,T)
    W_data_temp = W_data

    #Create our H matrix - this says where and how to combine our basis vectors
    H_data = np.random.rand(r,V_data_size[1])
    H_data_temp = H_data

    # #Now we run the update rule for number_of_iterations
    # W_row_sum = np.sum(W_data, axis=1)
    # W_col_sum = np.sum(W_data, axis=0)
    #
    # H_row_sum = np.sum(H_data, axis=1)
    # H_col_sum = np.sum(H_data, axis=0)

    k=0
    while k<number_of_iterations:

        V_over_lambda = np.divide(V_data, calculate_lambda(W_data, H_data, T))


        for t in range(0,T):

            H_mult_val = np.dot(np.transpose(W_data[..., ..., t]),shift_operator(V_over_lambda,-t))
            H_mult_val = np.divide(H_mult_val,np.dot(np.transpose(W_data[..., ..., t]),np.ones_like(np.transpose(W_data[..., ..., t]))))

            W_mult_val = np.dot(V_over_lambda,np.transpose(shift_operator(H_data,t)))
            W_mult_val = np.divide(W_mult_val,np.dot(np.ones_like(np.transpose(shift_operator(H_data,t))),np.transpose(shift_operator(H_data,t))))

            H_data_temp = np.multiply(H_data,H_mult_val)

            W_data_temp[..., ..., t] = np.multiply(W_data[..., ..., t],W_mult_val)

        k = k + 1
        W_data = W_data_temp
        H_data = H_data_temp



    return W_data, H_data, k;


X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])


conv_nmf(X,2,2,10)
