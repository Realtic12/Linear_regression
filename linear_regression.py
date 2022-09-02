#exercise linear regression by Class
#OK, can be used with other data

import matplotlib.pyplot as plt
import numpy as np

class Linear_Regression():
    def __init__(self,x,y):
        #define every parameter needed 
        self.x = x
        self.y = y

    def draw(self,y_pred): #plot the data

        repetir = True
        pregunta = 0
        #ask the user what size of house want
        print("Introduce el precio deseado a gastar entre 0 y 100000")
        while (repetir ==True):
            precio = int(input())
            if (0<precio<100000):
                repetir = False
            else:
                print("Introduce un valor correcto")

        #give a result for a single input by the user
        print(precio)
        pregunta = (precio/25000)*(self.w_act)+(self.b_act)
        print("El valor de la casa con {} sq² vale {}$".format(pregunta*25000,precio))

        plt.title("Price in USD related to Square feet") 
        plt.xlabel("Price ($USD)")
        plt.ylabel("Square feet")
        #plot the real data
        plt.scatter(self.x*25000, self.y*25000, label= 'Actual Data', color = 'g' )
        #Plot the predicted data
        plt.plot(self.x*25000,y_pred,label='Linear Regression', color = 'b')
        plt.legend()
        plt.show()


    def cost_final(self,w,b): #calculate the cost of the algorithm

        self.num_param = self.x.shape[0]
        cost = 0

        for i in range(self.num_param):
            f_wb = w * self.x[i] + b
            cost = cost + (f_wb - self.y[i])**2
            total_cost = 1 / (2 * self.num_param) * cost
        return total_cost

    def grad_descent (self,iterations,learn_rate): #calculate the gradien descent

        self.w_act = self.b_act = 0
        b_ant = w_ant = 0
        n=len(self.x)
        cost_actual = 0
        w_history = []
        b_history= []
        cost_ac =[]
        t=0

        for i in range (iterations):
            y_predicted =self.x*self.w_act+self.b_act
            cost_actual = self.cost_final(self.w_act,self.b_act)
            cost_ac.append(cost_actual)
            wd = -(2/n)*sum(self.x*(self.y-y_predicted))
            bd = -(2/n)*sum((self.y-y_predicted))
            self.w_act = self.w_act - learn_rate * wd
            self.b_act = self.b_act - learn_rate * bd
            w_history.append(self.w_act)
            b_history.append(self.b_act)

        return self.w_act,self.b_act,cost_actual,i

    def predict (self, X):
        return (X.dot(self.w_act)+self.b_act)

     
  
def main() :
      

    #reduce had been useful in order to scale features
    Y = np.array([0.0032, 0.0036, 0.004, 0.0048, 0.006, 0.0064, 0.008])      #Squarefeet (sq²)
    X = np.array([0.52 , 0.56 , 0.6 , 0.68 , 0.8 , 0.84, 1])      #Price ($USD)

    
    #convert the reduced data into the original
    x_origin = np.empty_like(X)
    y_origin = np.empty_like(Y)
    for t in range (X.shape[0]):
        x_origin[t] = X[t] * 25000
        y_origin[t] = Y[t] * 25000
    
    #data to test the parameters
    x_test = np.array([22000,21000,15500,18080,12300,15070,25000])

    #necessary parameters for the learning
    iterations = 100000
    learn_rate = 0.001
    ml = Linear_Regression(X,Y)
    w,b,cost,i=ml.grad_descent(iterations,learn_rate) #return the parameters w and b the actual cost and the iteration
    
    print("w: {},b: {}, cost: {},iteration: {}".format(w,b,cost,i))
    print("\n")

    Y_pred = ml.predict(X)

    ml.draw(Y_pred*25000)

    print(Y_pred*25000)
      
     
if __name__ == "__main__" : 
      
    main()

    
    