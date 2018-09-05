class foldwise:
    def __init__(self):
        self.trainTT=0
        self.k_fold=10
        self.accTra,self.accTes,self.precTraPos,self.precTraNeg,self.recTraPos,self.recTraNeg,self.precTesPos,self.precTesNeg,self.recTesPos,self.recTesNeg=(0 for i in range(10))
        self.train_TT,self.train_TF,self.train_FF,self.train_FT,self.test_TT,self.test_TF,self.test_FF,self.test_FT=(0 for i in range(8))
    def valid(self,z,cost,flag):
        if flag==0:
            if cost==0:
                if z==1:
                    self.train_TT+=1
                else:
                    self.train_FF+=1
            else:
                if z==0:
                    self.train_TF+=1
                else:
                    self.train_FT+=1
        else:
            if cost==0:
                if z==1:
                    self.test_TT+=1
                else:
                    self.test_FF+=1
            else:
                if z==0:
                    self.test_TF+=1
                else:
                    self.test_FT+=1

        
    def printfoldresult(self):
        eps=0
        # print("trainTT:",self.train_TT," trainTF:",self.train_TF," trainFF:",self.train_FF," trainFT:",self.train_FT)
        # print("train_accuracy",(self.train_TT+self.train_FF)/(self.train_TT+self.train_TF+self.train_FF+self.train_FT+eps))
        # print("train Precision +:",(self.train_TT)/(self.train_TT+self.train_FT+eps))
        # print("train Precision -:",(self.train_FF)/(self.train_FF+self.train_TF+eps))
        # print("train recall +:",(self.train_TT)/(self.train_TT+self.train_TF+eps))
        # print("train recall -:",(self.train_FF)/(self.train_FF+self.train_FT+eps))
        print("testTT:",self.test_TT," testTF:",self.test_TF," testFF:",self.test_FF," testFT:",self.test_FT)
        print("test_accuracy",(self.test_TT+self.test_FF)/(self.test_TT+self.test_TF+self.test_FF+self.test_FT+eps))
        print("test Precision +:",(self.test_TT)/(self.test_TT+self.test_FT+eps))
        print("test Precision -:",(self.test_FF)/(self.test_FF+self.test_TF+eps))
        print("test recall +:",(self.test_TT)/(self.test_TT+self.test_TF+eps))
        print("test recall -:",(self.test_FF)/(self.test_FF+self.test_FT+eps))

    def averageresults(self):
        eps=0
        # self.accTra+=(self.train_TT+self.train_FF)/(self.train_TT+self.train_TF+self.train_FF+self.train_FT+eps)
        # self.precTraPos+=((self.train_TT)/(self.train_TT+self.train_FT+eps))
        # self.precTraNeg+=((self.train_FF)/(self.train_FF+self.train_TF+eps))
        # self.recTraPos+=((self.train_TT)/(self.train_TT+self.train_TF+eps))
        # self.recTraNeg+=((self.train_FF)/(self.train_FF+self.train_FT+eps))
        self.accTes+=((self.test_TT+self.test_FF)/(self.test_TT+self.test_TF+self.test_FF+self.test_FT+eps))
        self.precTesPos+=((self.test_TT)/(self.test_TT+self.test_FT+eps))
        self.precTesNeg+=((self.test_FF)/(self.test_FF+self.test_TF+eps))
        self.recTesPos+=((self.test_TT)/(self.test_TT+self.test_TF+eps))
        self.recTesNeg+=((self.test_FF)/(self.test_FF+self.test_FT+eps)) 

    def reset(self):
        self.train_TT,self.train_TF,self.train_FF,self.train_FT,self.test_TT,self.test_TF,self.test_FF,self.test_FT=(0 for i in range(8))
        
    def printaverageresults(self):
        print("average result for ",self.k_fold,"fold")
        # print("Last fold confusion matrix trainTT:",self.train_TT," trainTF:",self.train_TF," trainFF:",self.train_FF," trainFT:",self.train_FT)
        # print("Last fold testTT:",self.test_TT," testTF:",self.test_TF," testFF:",self.test_FF," testFT:",self.test_FT)
        # print("train_accuracy:",self.accTra/self.k_fold)
        # print("train Precision +:",self.precTraPos/self.k_fold)
        # print("train Precision -:",self.precTraNeg/self.k_fold)
        # print("train recall +:",self.recTraPos/self.k_fold)
        # print("train recall -:",self.recTraNeg/self.k_fold)
        print("test_accuracy",self.accTes/self.k_fold)
        print("test Precision +:",self.precTesPos/self.k_fold)
        print("test Precision -:",self.precTesNeg/self.k_fold)
        print("test recall +:",self.recTesPos/self.k_fold)
        print("test recall -:",self.recTesNeg/self.k_fold)
