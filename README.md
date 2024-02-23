# Q_learning
 A q-learning utilisation

 --------------- Master Thesis ---------------
 
 ---------------------------------------------
 
 ---------------------------------------------
 
 -------------- A Case Study of -------------- 
 
 --- Markov Decision Processes & Q-Learning --
 
 ---------- under Model Uncertainty ----------
 
 ------------- in finite spaces --------------

 -------------------- & ----------------------

 --------- a practical application -----------

 ------------- on Fishing Quotas -------------

 Carried out with NUS

 Supervisor: Julian Sester
 
 Reference: https://github.com/juliansester/Wasserstein-Q-learning.git



 Implementation of basic q-learning functions:
 
    - Non-robust and Robust
    
    - Focusing on our finite spaces case
    
    - Examples implementation



 Important things to note:
 
    - The q_learning and robust_q_learning files are the algorithms implemented and ready to use. In order to use them you just have to correctly define the inputs, see the jupyter notebook files for examples (NonRobust_cointoss, Robust_cointoss, FishingQuotas)
    
    - The robust_q_learning_v2 is simply the first version of the algorithm in finite space made more effective, because an indicator fonction was removed allowing the algorithm to update the matrix efficiently. It is a way to jump all the steps for which the Q_t matrix is not updated, and jump directly to those where there is an updating.
    
    - The robust_q_learning_v3 and q_learning_v2 are the algorithms that can now be applied to cases that are in practice non-recurrent Markov chains, still need to be proved.
