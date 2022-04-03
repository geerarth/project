import pandas as pd 
import numpy as np, random


##################################################################################

def get_security_transition_matrix():
    # The transition matrix is the same regardless if the game is cycled or not
    # /!\ index different from square number (need to do -1)
    n_squares = 15 
    T_m = np.zeros((n_squares, n_squares))
    #i = index from ==> on ne commence jamais à cet index car on gagné à ce moment. La proba =1. C'est mis après 
    for i in range(n_squares-1): 
        T_m[i][i] = 1/2
        if(i==2):
            T_m[i][i+1] = 1/4 #if takes slow line
            T_m[i][10] = 1/4 #if takes fast line
        elif(i==9):
            T_m[i][14] = 1/2            
        else:
            T_m[i][i+1] = 1/2
    T_m[14][14] = 1
    return T_m

def get_normal_transition_matrix(layout, cycle:bool):
    # The transition matrix is the same regardless tif the game is cycled or not
    # If the player just passes through square 3 without stopping, 
    # he continues to square 4 or beyond (5, 6, etc), as if the other path does not exist
    n_squares = 15 
    T_m = np.zeros((n_squares, n_squares))
    for i in range(n_squares-1): # o
        T_m[i][i] = 1/3
        if(i==2):
            # #if takes slow line
            T_m[i][i+1] = 1/6 
            T_m[i][i+2] = 1/6
            #if takes fast line
            T_m[i][10] = 1/6
            T_m[i][11] = 1/6

        elif(i==8):
            T_m[i][i+1] = 1/3
            T_m[i][14] = 1/3

        elif(i==9 or i==13):
            if(cycle): 
                T_m[i][14] = 1/3
                T_m[i][0] = 1/3
            else:
                T_m[i][14] = 2/3

        else:
            T_m[i][i+1] = 1/3
            T_m[i][i+2] = 1/3
    T_m[n_squares-1][n_squares-1] = 1

    
    # Il faut encore prendre les pièges et bonuses en compte 
    #il faut reparcourir Chaque élément de la T_m
    for i in range(n_squares-1): # on ne veut pas changer le point de départ ni le point d'arrivé car il n'y a ni piège 
        for j in range(n_squares-1):
                # une chance sur 2 de trigger le piège ou bonus
                if(layout[j]==1 and j!=0): # type 1: go back at 1st square
                    T_m[i][0] += T_m[i][j]/2 # il faut transférer une demi proba vers le square 1
                    T_m[i][j] /=2 # une fois le transfert de proba fait, il faut bien l'enlever en [i][j] forcément
                elif (layout[j]==2 and j!=0): #type 2: go back 3 squares
                    #il y a qlq cas spéciaux
                    if (j>=10 and j <=12): # if j ==11 12 or 13 --> go to 1,2 or 3 with proba 1/2
                        #diff = 10
                        T_m[i][j-10] += T_m[i][j]/2
                        T_m[i][j] /=2 # une fois le transfert de proba fait, il faut bien l'enlever en [i][j] forcément
                        
                    # si trap type 2 à la case 2 ou 3, on peut pas aller moins loin que le start 
                    elif(j<3):
                        T_m[i][0] += T_m[i][j]/2 # il faut transférer une demi proba vers le square 1
                        T_m[i][j] /=2 # une fois le transfert de proba fait, il faut bien l'enlever en [i][j] forcément
                        
                    else: # cas "commun"
                        T_m[i][j-3] += T_m[i][j]/2
                        T_m[i][j] /=2 # une fois le transfert de proba fait, il faut bien l'enlever en [i][j] forcément
                
    return T_m

def get_risky_transition_matrix(layout, cycle:bool):
    
    n_squares = 15
    T_m = np.zeros((n_squares, n_squares))

    for i in range(n_squares-1): # on prend le dernier square après la boucle, d'où le -1

        T_m[i][i] = 1/4
        if(i==2):
            # #if takes slow line
            T_m[i][i+1] = 1/8 
            T_m[i][i+2] = 1/8
            T_m[i][i+3] = 1/8
            #if takes fast line
            T_m[i][10] = 1/8
            T_m[i][11] = 1/8
            T_m[i][12] = 1/8
        
        elif(i==7):
            T_m[i][i+1] = 1/4
            T_m[i][i+2] = 1/4
            T_m[i][14] = 1/4 #(du 10 au 15)

        elif(i==8 or i==12):
            T_m[i][9] = 1/4
            if cycle :
                T_m[i][14] = 1/4
                T_m[i][0] = 1/4
            else : 
                T_m[i][14] = 1/2
        
        elif(i==9 or i == 13):
            if cycle :
                T_m[i][14] = 1/4
                T_m[i][0] = 1/4
                T_m[i][1] = 1/4
            else :
                T_m[i][14] = 3/4
        
        else : 
            T_m[i][i+1] = 1/4
            T_m[i][i+2] = 1/4
            T_m[i][i+3] = 1/4
    
    T_m[n_squares-1][n_squares-1] = 1
    
    
    # LES PIEGES :
    for i in range(n_squares-1): # on ne veut pas changer le point de départ ni le point d'arrivé car il n'y a ni piège 
        for j in range(n_squares-1):
            # Le trap est d'office triggered si c'en est un
            if(layout[j]==1 and j!=0): # type 1: go back at 1st square # J'ai ajouté and j!=! pcq ca rajoutait la condition pour soit même.
                T_m[i][0] += T_m[i][j]
                T_m[i][j] = 0 # une fois le transfert de proba fait, il faut bien l'enlever en [i][j] forcément

            elif(layout[j]== 2 and j!=0): # -3 squares
                if (j>=10 and j <=12):
                        #diff = 10
                    T_m[i][j-10] += T_m[i][j]
                    T_m[i][j] = 0 # une fois le transfert de proba fait, il faut bien l'enlever en [i][j] forcément
                # si trap type 2 à la case 2 ou 3, on peut pas aller moins loin que le start 
                elif(j<3):
                    T_m[i][0] += T_m[i][j] # il faut transférer une demi proba vers le square 1
                    T_m[i][j] = 0 # une fois le transfert de proba fait, il faut bien l'enlever en [i][j] forcément
                else: # cas "commun"
                    T_m[i][j-3] += T_m[i][j]
                    T_m[i][j] = 0 # une fois le transfert de proba fait, il faut bien l'enlever en [i][j] forcément
            

                    

    return T_m


#########################################################################################################""

class SimulateAGame():
    def __init__(self, policy, layout, circle):
        self.layout =layout
        self.circle = circle
        self.policy = policy
        self.cost = 0

    def get_cost_game(self):
        """
        policy says which dice to take via int number: 
        - 1 : security
        - 2 : normal 
        - 3 : risky
        """
        self.cost = 0
        return self.get_cost_from(0)
    
    def get_cost_from(self, index_from):
        """ 
        """
        #assert(len(policy) == len(self.layout))
        current_index = index_from 
        while(current_index != 14):
            dice = self.policy[current_index] #{1,2,3}
            if(dice not in [1,2,3]): raise ValueError('sorry zobi')
            current_index = self.play(current_index, dice)
            self.cost+=1    
        return self.cost

    def play(self, current_index, dice_type):
        """return resulting index"""        
        if( dice_type == 1):  # no traps nor bonuses
            if(current_index == 2): return np.random.choice([2,3,10], p = [0.5, 0.25, 0.25] )
            elif (current_index == 9): return np.random.choice([9,14] )
            else: return np.random.choice([current_index, current_index+1])
        elif(dice_type ==2): # normal dice: 1/2 to trigger trap/bonus
            is_triggered = np.random.choice([True,False]) #50% chance to trigger trap/bonus

            if(current_index==2): next_index = np.random.choice([current_index,3,4,10,11], p = [1/3, 1/6, 1/6, 1/6, 1/6])
            elif(current_index==8): next_index = np.random.choice([current_index,9,14])
            elif(current_index==9 or current_index == 13):
                if(self.circle): 
                    next_index = np.random.choice([current_index,14,0])
                else: next_index = np.random.choice([current_index,14], p = [1/3,2/3])

            # Ne faut il pas ajouté pour tous les autres cas ?
            else : next_index = np.random.choice([current_index, current_index+1, current_index+2])
                
            if (is_triggered and next_index!=14): return self.activateTrapOrBonus(next_index)
            else: return next_index

        else: # type =3, i.e. risky
            if(current_index==2):
                next_index = np.random.choice([current_index,3,4,5, 10,11,12], p = [1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
            elif(current_index==7):
                next_index = np.random.choice([current_index,8,9,14])
            elif(current_index==8):
                if(self.circle): next_index = np.random.choice([current_index,9,14,0])
                else: next_index = np.random.choice([current_index,9,14], p = [1/4, 1/4, 2/4])
            elif(current_index==9):
                if(self.circle): next_index = np.random.choice([current_index,14,0,1])
                else: next_index = np.random.choice([current_index,14], p = [1/4, 3/4])
            elif(current_index==12):
                if(self.circle): next_index = np.random.choice([current_index,13,14,0])
                else: next_index = np.random.choice([current_index,13,14], p = [1/4, 1/4, 2/4])
            elif(current_index==13):
                if(self.circle): next_index = np.random.choice([current_index,14,0,1])
                else: next_index = np.random.choice([current_index,14], p = [1/4, 3/4])
            
            else: next_index = np.random.choice([current_index, current_index+1 , current_index+2, current_index+3])

            return self.activateTrapOrBonus(next_index)

    def activateTrapOrBonus(self,index):
        if (self.layout[index] == 1): return self.activateType1Trap(index)
        elif (self.layout[index] == 2): return self.activateType2Trap(index)
        elif (self.layout[index] == 3): return self.activateType3Trap(index)
        elif (self.layout[index] == 4): return self.activateBonus(index)
        else: return index # ordinary square (0)

    def activateType1Trap(self, index):
        if (index!=14):
            return 0
        else : return index


    def activateType2Trap(self, index):
        if (index!=14):
            if(index<3 or index == 10): 
                return 0 # or index = 10 je crois
            elif(index == 11 or index == 12): 
                return (index-10) # Entre 11 et 12 et non entre 10 et 12
            else: 
                return (index-3)
        else : return index

    def activateType3Trap(self, index):
        if (index!=14):
            self.cost += 1
        return index

    def activateBonus(self, index):
        if (index!=14):
            self.cost -= 1
        return index
    

    def get_policy_history(self, policy= None):
        if policy is None: policy = self.policy # si on ne met rien dedans, ça met la policy de base. Mais on peut voir ce que ça fait selon une politique donnée 
        history = pd.DataFrame(columns= ["Nb_of_throws", "action", "current_index", "layout_info" ])
        current_index = 0 
        current_cost = 0
        while(current_index != 14):
            dice = policy[current_index] #{1,2,3}
            if(dice not in [1,2,3]): raise ValueError('sorry zobi')
            # Adding information to history:
            history.loc[len(history.index)] = [current_cost, dice, current_index, self.layout[current_index] ]
            current_index = self.play(current_index, dice)
            #update number of throws
            current_cost+=1    
            
        return history


#########################################################################################################################

class MarkovDecision():
    def __init__(self, layout:  np.ndarray, cycle:bool):
        self.security_transition_matrix = get_security_transition_matrix()
        self.normal_transition_matrix = get_normal_transition_matrix(layout,cycle)
        self.get_risky_transition_matrix = get_risky_transition_matrix(layout,cycle)

        self.layout = layout
        self.cycle = cycle
        self.dices = [1,2,3]
        self.states = range(15)

        self.board = SimulateAGame(policy=None, layout= layout, circle=cycle)
        
    
    # at each step, one needs to choose which dice, which means which transitions matrix on which to play in order to minimise the expected number of tosses
    
    def compute_expected_cost_next_state(self, index: int, V_hat: np.array):
        """
        # v_k = current_state proba vector
        # return next state proba vector
        """

        if (index in list(range(0,15))): #if index in game board 
            secur = np.dot(self.security_transition_matrix[index], V_hat)
            normal = np.dot(self.normal_transition_matrix[index], V_hat)
            risky = np.dot(self.get_risky_transition_matrix[index], V_hat)
            return(secur, normal, risky)


    def run(self, max_epoch= 1000):
        """ 
        return Expec (length = 14) and Dice with the optimal dice choice
        """
        # Set value iteration parameters
        delta = 1e-9 # Error tolerance
        V = np.append( (np.ones(14)*100),[0]) # Initialize values
        pi = np.zeros(15) # Initialize policy

        for i in range(max_epoch):
            #delta = 0 # Initialize max difference
            max_diff = 0  # Initialize max difference
            V_new = np.append( (np.ones(14)*100),[0]) # Initialize values

            for s in self.states:
                ## Compute state value

                secu_expected_cost = 1 + np.dot(self.security_transition_matrix[s], V)  
                norm_expected_cost = 1 + np.dot(self.normal_transition_matrix[s], V)  
                risk_expected_cost = 1 + np.dot(self.get_risky_transition_matrix[s], V)

                # il n'y a que les piège = 3 et les bonus qui font varier la fonction de cout. Les autres te font seulement changer de case, ce qui est pris en compte dans la transition matrix
                if(self.layout[s] == 3):
                    secu_expected_cost += 1 
                    norm_expected_cost += 1
                    risk_expected_cost += 1
                elif(self.layout[s]==4):
                    secu_expected_cost -= 1 
                    norm_expected_cost -= 1
                    risk_expected_cost -= 1

                #print(risk_expected_cost)

                # Store value best action so far
                #print("for state " + str(s) + ": " + str([secu_expected_cost, norm_expected_cost,risk_expected_cost]))
                costs_to_compare = [secu_expected_cost, norm_expected_cost,risk_expected_cost]
                min_cost = min(costs_to_compare)
                if(s==14): min_cost = 0 # au dernier state, c'est fini, on a gagné :) 
                V_new[s] = min_cost  # Update value with lowest value

                # Update best policy
                best_index = np.argmin(costs_to_compare)
                best_action = best_index +1 
                if(s!=14):
                    pi[s] =  best_action # Store action with highest value

                # Update maximum difference
                max_diff = max(max_diff, abs(V[s] - V_new[s]))

            # Update value functions
            V = V_new

            # If diff smaller than threshold delta for all states, algorithm terminates
            #print(max_diff)
            if max_diff < delta:
                print("algo has converged after " + str(i) + " eopchs" )
                return V,pi
        print("algo reached maximal number of epochs without converging")
        return V, pi


"""def get_proba_next_move_position(v_k, T_m):
    # v_k = current_state proba vector
    # return next state proba vector
    return np.dot(v_k, T_m)"""



def markovDecision(layout, circle):
    mdp = MarkovDecision(layout, circle)
    return mdp.run()

markovDecision(layout=[0,2,3,0,0,3,2,0,2,3,0,0,3,0,0], cycle=False)