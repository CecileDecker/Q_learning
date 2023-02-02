
# Classical Q learning #

def q_learning(X,
               A,
               r,
               P_0, # Simulation of next state in dependence of x and a
               alpha,
               x_0, 
               eps_greedy = 0.05,
               Nr_iter = 1000,
               gamma_t_tilde = lambda t: 1/(t+1),
               Q_0 = None):
    """
    Parameters
    ----------
    X : numpy.ndarray
        A list or numpy array containing all states
    A : numpy.ndarray
        A list or numpy array containing all actions
    r : function
        Reward function r(x,a,y) depending on state-action-state.
    P_0 : function
        fucntion P_0(x,a) that creates a new random variabe in dependence of state and action
    alpha : float
        Discounting rate.
    x_0 : numpy.ndarray
        the initial state.
    eps_greedy : float, optional
        Parameter for the epsilon greedy policy. The default is 0.05.
    Nr_iter : int, optional
        Number of Iterations. The default is 1000.
    gamma_t_tilde : function, optional
        learning rate. The default is lambda t: 1/(t+1).
    Q_0 : matrix, optional
        Initial value for the Q-value matrix. The default is None.

    Returns
    -------
    matrix
        The Q value matrix.
    """
    rng = np.random.default_rng()    
    #Initialize Q_0
    Q = np.zeros([len(X),len(A)])
    if Q_0 is not None:
        Q = Q_0
    Visits = np.zeros([len(X),len(A)])
    if np.ndim(A)>1:
        A_list = A
    else:
        A_list = np.array([[a] for a in A])
    if np.ndim(X)>1:
        X_list = X
    else:
        X_list = np.array([[x] for x in X])

    def a_index(a):
        return np.flatnonzero((a==A_list).all(1))[0]
    def x_index(x):
        return np.flatnonzero((x==X_list).all(1))[0]
    
    # Define the f function
    def f(t,x,a,y):
        return r(x,a,y)+alpha*np.max(Q[x_index(y),:])
  
    def a_t(t,y):
        #eps_bound = 1-(t/Nr_iter)*(eps_greedy)
        eps_bound = eps_greedy
        unif = np.random.uniform(0)
        return (unif>eps_bound)*A[np.argmax(Q[x_index(y),:])]+(unif<=eps_bound)*rng.choice(A)
        
    
    # Set initial value        
    X_0 = x_0
    # List of differences of Q Matrices 
    for t in tqdm(range(Nr_iter)):
        X_1 = P_0(X_0,a_t(t,X_0))
        Q_old = copy.deepcopy(Q)
        x,a = X_0, a_t(t,X_0)
        x_ind, a_ind = x_index(x),a_index(a)
        # Do the update of Q        
        Q[x_ind, a_ind] = Q_old[x_ind, a_ind]+gamma_t_tilde(Visits[x_ind, a_ind])*(f(t,x,a,X_1)-Q_old[x_ind, a_ind])
        Visits[x_ind, a_ind]+=1
        X_0 = X_1
    return Q
