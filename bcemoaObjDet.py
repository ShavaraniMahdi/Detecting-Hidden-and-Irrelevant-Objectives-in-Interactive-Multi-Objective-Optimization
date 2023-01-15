#importing libraries
import numpy as np
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from svmrank import utility2prefs, RankSVM
from sklearn.svm import SVR
from copy import copy,deepcopy

# MANUEL: This should not be necessary.
import problems
from problems import as_dummy
from ea_operators import tournament_selection, polynomial_mutation, \
    sbx_crossover, select_best_N_mo

'''
*

* @param[in] cr Crossover probability.
* @param[in] eta_c Distribution index for crossover.
* @param[in] m Mutation probability.
* @param[in] eta_m Distribution index for mutation.
* @param seed seed used by the internal random number generator (default is random)
* @throws std::invalid_argument if \p cr is not \f$ \in [0,1[\f$, \p m is not \f$ \in [0,1]\f$, \p eta_c is not in
* [1,100[ or \p eta_m is not in [1,100[.
* @param gen1: number of nsga2 generations before the first interaction of bcemoa
* @param geni: number of generations between two consecutive interactions
* @param sampleSize: number of solutions presented to the DM for pairwise comparison
* @param detection:  determines if the detection of hidden objectives and consequently the update of objs are active or not                                   
*/'''    
#Main BCEMOA Class
#Defining bcemoa class based on nsga2 class in Pygmo
class bcemoa():
    def __init__ (self, dm, total_gen=500, gen1 = 100, cr = 0.95, eta_c = 10., m = 0.01, eta_m = 50.,
                  seed = 564654561, geni=20, interactions=5, sampleSize=5, verbose = 0, detection =0, idi=None, archive=False, threshold=None, recursive=False):
        np.random.seed(seed)
        self.nsga2 = pg.algorithm(pg.nsga2(gen=gen1, cr=cr, eta_c=eta_c, m=m, eta_m=eta_m, seed=seed))
        self.geni = geni
        self.interactions=interactions
        self.mdm = dm
        self.m_cr = cr
        self.m_m = m
        self.m_eta_c = eta_c
        self.m_eta_m = eta_m
        self.sampleSize = sampleSize
        self.m_log = []
        self.total_gen=total_gen
        self.verbose = verbose
        self.detection= detection
        self.idi=idi
        self.archive=archive
        self.threshold=threshold
        self.recursive=recursive
        
        if self.idi!=None:
            self.identified_df=pd.DataFrame(columns=['idi','interaction','vf' ,'dv','f', 'Mode', 'Interactions', 'Problem', 'SVM_score','a'])
    def get_name(self):
        return 'BCEMOA_Detection'    
    # Overloading evolve function of nsga2   
    def evolve(self,pop):
        # prob=pop.problem
        # temp_dv=as_dummy(prob).get_dv()
        # as_dummy(prob).set_dv(np.arange(len(as_dummy(prob).original_fitness(pop.get_x()[0]))))
        pop = self.nsga2.evolve(pop)
        # as_dummy(prob).set_dv()
        pop = self.evolvei(pop)
        return pop

    #defining the evolvei function in bcemoa used in evolve
    #evolvei uses the ranking of the solutions returned by SVM
    #To priorize the solutions instead of using non-domination sorting        
    def evolvei(self, pop):
        if self.idi!=None:
            ff=as_dummy(pop.problem).original_fitness(pop.get_x()[0])
            dv =as_dummy(pop.problem).get_dv()
            df2=pd.DataFrame([[self.idi, 0, self.mdm.value(ff,mode=1), dv,ff, self.mdm.mode , self.interactions, pop.problem.get_name(), '-', len(dv)]],columns=['idi','interaction','vf', 'dv','f', 'Mode', 'Interactions', 'Problem', 'SVM_score', 'a'])
            self.identified_df=self.identified_df.append(df2, ignore_index=True, sort=False)
        N = len(pop)
        if N < self.sampleSize:
            print(f"Warning: Sample size={self.sampleSize} was larger than N={N}, adjusting it")
            self.sampleSize = N
                
        prob = pop.problem
        # original_f = as_dummy(prob).get_original_f(pop)
        # We use scaler to scale all features to the interval (0,1).
        # We fit the data to the entire pop so that the scaled values stay in the range (0,1).
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # The scaler is trained on the original fitness, otherwise we will have
        # wrong range information.
        # scaler.fit(original_f)
       
        # A list of fitness vectors for training
        training_set=[]
        training_x=[]
        rank_pref = np.empty(0)
        pairwise= np.empty(0)
        if self.archive:
            archive=pg.population(prob)
        for LearningIteration in range(self.interactions):
            #####################################################
            # Interaction and Learning  
            #####################################################
            
            training_set, training_x, rank_pref, pairwise= self.get_preferences(pop,training_set,training_x, rank_pref, pairwise)
            
            self.training_record=pd.DataFrame(columns=['idi','interaction','objective_vector','ranks','problem','Mode'])
            if self.detection:
                pop = update_pop(pop, training_set, rank_pref, self.verbose, threshold=self.threshold, recursive=self.recursive)
                prob=pop.problem
            if self.mdm.mode != 1: # rank the model by svm;   
                predictor=learn(np.asarray(training_set)[:,as_dummy(prob).get_dv()], pairwise, self.verbose)
                pref_fun = predictor.predict 
                # pref_fun= lambda x: predictor.predict(MinMaxScaler().fit_transform(x))
            else:# when the mode is 1 no learning is done
                pref_fun = self.mdm.value_array
                
            ######################################################
            # bcemoa iterations: geni iterations after each interaction  
            ######################################################
            if LearningIteration == self.interactions-1:
                self.geni = self.total_gen - (self.interactions - 1) * self.geni
            # The NSGA2 loop with preference replacing the crowding distance
            for gens in range(self.geni):
                if self.mdm.mode==1:
                    f = as_dummy(prob).get_original_f(pop)
                else:
                    f = pop.get_f()
                pop_pref = pref_fun(f)   
                _, _, _, ndr = pg.fast_non_dominated_sorting(f)
                shuffle1 = np.random.permutation(N)
                shuffle2 = np.random.permutation(N)
                x = pop.get_x()
                # We make a shallow copy to not change the original pop.
                pop_new = deepcopy(pop)
                for i in range(0, N, 4):
                    child1, child2 = self.generate_two_children(prob, shuffle1, i, x, ndr, -pop_pref)
                    # we use prob to evaluate the fitness so that its feval
                    # counter is correctly updated
                    pop_new.push_back(child1, prob.fitness(child1))
                    pop_new.push_back(child2, prob.fitness(child2))
                    
                    child1, child2 = self.generate_two_children(prob, shuffle2, i, x, ndr, -pop_pref)
                    pop_new.push_back(child1, prob.fitness(child1))
                    pop_new.push_back(child2, prob.fitness(child2))

        
                # Selecting Best N individuals
                if self.mdm.mode==1:#To use the true utility value we need to have complete answeres in the experiments
                #to calculate UF when mode=1.
                    f = as_dummy(prob).get_original_f(pop)
                else:
                    f = pop.get_f()
                best_idx = select_best_N_mo(f, N, pref_fun)
                                
                assert len(best_idx) == N
                x = pop_new.get_x()[best_idx]
                f = pop.get_f()[best_idx]
                for i in range(len(pop)):
                    pop.set_xf(i, x[i], f[i])
                
                if self.verbose:
                    print(f"best vf: {self.mdm.pref.value(as_dummy(prob).get_original_f(pop)[0])}")
                    # plt.clf()
                    # fig = plt.figure(figsize = (10, 7)) 
                    # ax =plt.axes(projection ="3d") 
                    # ax.scatter3D(scaled_af[:,0], scaled_af[:,1], scaled_af[:,2])
                    # plt.scatter(pop.get_f()[:,0], pop.get_f()[:,1])
                    # plt.xlim(0.0, None)
                    # plt.ylim(0.0, None)
                    # plt.pause(0.000001)
            if self.idi!=None:
                dv =as_dummy(pop.problem).get_dv()
                ff=as_dummy(prob).original_fitness(training_x[np.where(rank_pref==0)[0][-1]])
                df2=pd.DataFrame([[self.idi, LearningIteration+1, self.mdm.value(ff,mode=1), dv,ff, self.mdm.mode , self.interactions, pop.problem.get_name(), '-',len(dv)]],columns=['idi','interaction','vf', 'dv','f', 'Mode', 'Interactions', 'Problem', 'SVM_score' ,'a'])
                self.identified_df=self.identified_df.append(df2, ignore_index=True,sort=False)
        
        if self.idi!=None:
            data={'idi':self.idi,'interactions':self.interactions,'objective_vector':training_set,'ranks':rank_pref,'problem':pop.problem.get_name(),'Mode':self.mdm.mode}
            self.training_record=pd.DataFrame(data)
            
            filename = "training_set-{}-".format(self.idi) + ".csv"
            self.training_record.to_csv(filename, index=False)
        if self.archive:
            training_set, training_x, rank_pref, pairwise= self.get_preferences(pop,training_set,training_x, rank_pref, pairwise)
            for sol in range(self.sampleSize):
                archive.push_back(training_x[-(sol+1)])
            f = as_dummy(prob).get_original_f(archive)
            best_idx = select_best_N_mo(f, len(archive), self.mdm.value_array)
            x = archive.get_x()[best_idx]
            archive_new=pg.population(prob)
            for i in range(len(archive)):
                archive_new.push_back(x[i])
            pop=archive_new
        if self.idi!=None:
            dv =as_dummy(pop.problem).get_dv()
            ff=as_dummy(prob).original_fitness(training_x[np.where(rank_pref==0)[0][-1]])
            df2=pd.DataFrame([[self.idi, LearningIteration+1, self.mdm.value(ff,mode=1), dv,ff, self.mdm.mode , self.interactions, pop.problem.get_name(), '-', len(dv)]],columns=['idi','interaction','vf', 'dv','f', 'Mode', 'Interactions', 'Problem', 'SVM_score', 'a'])
            self.identified_df=self.identified_df.append(df2, ignore_index=True,sort=False)
            filename = "identified-{}-".format(self.idi) + ".csv"
            self.identified_df.to_csv(filename, index=False)
        return pop

    def get_preferences(self, pop, training_set, training_x, rank_pref, pairwise):
        prob=pop.problem
        # This function assumes that pop is already sorted by non-dominated
        # sorting breaking ties with the predicted utility function.

        if len(rank_pref)==0: #FIXME: The condition is added to include the selected solution in the previous interaction
            remaining = self.sampleSize
        else:
            training_set.append(training_set[np.where(rank_pref==0)[0][-1]])
            training_x.append(training_x[np.where(rank_pref==0)[0][-1]])
            remaining = self.sampleSize-1
        
        # MANUEL: Remove duplicates is counter-productive when you may have
        # hidden objectives because the DM behaving different for solutions
        # that look the same is helpful to detect hidden objectives.
        # Mahdi: Fixed
        
        # for i in range(len(pop)):
        #     f = as_dummy(prob).original_fitness(pop.get_x()[i])
        #     if len(training_set)==0 : #or not np.any(np.all(np.isclose(training_set,f),axis=1))
        #         training_set.append(f)
        #         remaining -= 1
        #         if remaining == 0:
        #             break
        first=len(training_set)==0
        indexes= range(remaining)
        if first:
            ndf, _, _, ndr = pg.fast_non_dominated_sorting(pop.get_f())
            if len(ndf[0])>remaining:
                indexes=np.random.choice(ndf[0], size=remaining, replace=False)
        for i in indexes:
            f = as_dummy(prob).original_fitness(pop.get_x()[i])
            training_x.append(pop.get_x()[i])
            training_set.append(f)

            
        start = len(rank_pref)
        # This is where the interaction with the DM occurs.
        tmp_rank_pref = self.mdm.setRankingPreferences(training_set[start:])
        return training_set, training_x, np.append(rank_pref, tmp_rank_pref).astype(int), np.append(pairwise, utility2prefs(tmp_rank_pref)+start).reshape(-1,2).astype(int)

    
    def crossover(self, problem, parent1, parent2):
        return sbx_crossover(problem, parent1, parent2, self.m_cr, self.m_eta_c)
      
    def mutate(self, problem, child):
        return polynomial_mutation(problem, child, self.m_m, self.m_eta_m)
                          
    def generate_two_children(self, problem, shuffle, i, X, ndr, pop_pref):
        parent1_idx = tournament_selection(shuffle[i], shuffle[i + 1], ndr, pop_pref) 
        parent2_idx = tournament_selection(shuffle[i + 2], shuffle[i + 3], ndr, pop_pref)
        parent1 = X[parent1_idx]
        parent2 = X[parent2_idx]
        child1, child2 = self.crossover(problem, parent1, parent2)
        child1 = self.mutate(problem, child1)
        child2 = self.mutate(problem, child2)
        return child1, child2

    def get_log(self):
        return self.m_log
    
def update_pop(pop, training_set, rank_pref, verbosity, threshold=None, recursive=None):
    prob=pop.problem
    dv =as_dummy(prob).get_dv()
    
    #if threshold is not given, length of dv is not changed
    #if threshold is given, factors with pvalue<thereshold are selected, if 
    # number of objs with pvalues<threshold is less than 2, two most important ones are selected.
    if threshold==None:
        if recursive:
            estimator = SVR(kernel="linear")
            selector = RFE(estimator, n_features_to_select=len(dv), step=1)
        else:
            selector = SelectKBest(f_regression, k = len(dv))#
        
        
        # We use feature_selection.selectKBest in sklearn to select
        # most important objs and check if the existing dv vector
        # is up to date
        selector.fit(training_set, rank_pref) # we use preprocessing-scale as most of learning algorithm assume features with mean zero and unit variance. SRC:https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
        new_dv = selector.get_support(True)
    else:
        if recursive:
            selector = SelectFromModel(estimator=LogisticRegression()).fit(training_set, rank_pref)
            new_dv=selector.get_support(indices=True)
        else:
            selector = SelectKBest(f_regression, k = len(training_set[0]))
            selector.fit(training_set, rank_pref)
            n=sum(selector.pvalues_<=threshold)
            if n<2:
                new_dv=selector.pvalues_.argsort()[:2]
            else:
                new_dv=selector.pvalues_.argsort()[:n]
    # If selected features are different than dv, we update pop and mdm vectors
    if set(new_dv) != set(dv):
        if verbosity:
            print(f"old dv: {dv} \nnew_dv is {new_dv}")
        as_dummy(prob).set_dv(new_dv)#Set_dv didn't update the population and hence there isproblem when dimenstion chages
        x = pop.get_x()
        original_f = as_dummy(prob).get_original_f(pop)
        f = original_f[:,new_dv]
            
        #####
        as_dummy(pop.problem).prob
        dummy_prob = problems.dummy(as_dummy(pop.problem).prob, new_dv)
        pop2 = pg.population(dummy_prob, size = len(pop))
        for i in range(len(pop)):
            pop2.set_xf(i, x[i], f[i])
        pop=pop2
    return pop
    
    

def learn(training_set, pairwise, verbose):
    # tuned_parameters = [ {'kernel': ['rbf','linear','poly' ], 'degree':[2,3], 'gamma': [10**x for x in list(range(-3,3))],'C': [1,10,100]}]
    # With sklearn 0.22.2.post1 kernel=poly,gamma=10**3,C=100 gets stuck, is it a bug?
    '''
    if len(training_set)>30:
        tuned_parameters = [
            {'kernel':['linear'], 'C':[1]},
            {'kernel':['rbf'],    'C':[1], 'gamma': 10.0**np.arange(-3,-1)}, 
            {'kernel':['poly'],   'C':[1], 'gamma': 10.0**np.arange(-3,-1),'degree':[2]}]
    else:
        tuned_parameters = [
            {'kernel':['linear'], 'C':[1,10,100]},
            {'kernel':['rbf'],    'C':[1,10,100], 'gamma': 10.0**np.arange(-3,1)}, 
            {'kernel':['poly'],   'C':[1,10,100], 'gamma': 10.0**np.arange(-3,1),'degree':[2]}]

    if len(training_set) < 4:
        learner = svm.SVR(kernel='linear', C=100, gamma='auto')
    else:
        cv = 3 if len(training_set) > 5 else 2
        learner = model_selection.GridSearchCV(svm.SVR(), tuned_parameters, cv=cv,
                                               verbose = verbose, n_jobs = -1, scoring='explained_variance')
    # print(f'length of training set is {len(training_set)}')
    start = time.time()
    # learner.fit(training_set[-10:], rank_pref[-10:])
    tr=MinMaxScaler().fit_transform(training_set)
    ra=rank_pref*1
    ra=ra=(ra-min(ra))/(max(ra)-min(ra))
    learner.fit(tr, ra)
    end=time.time()
    # print(f'the duration for fitting was {end-start}')
    if verbose >= 1:
        print(learner)
        print(learner.best_params_)
        #print(learner.cv_results_)
    print(f'best score: {learner.best_score_}')
    '''    
    #Performing GridSearch and returning best estimator by ranksvm
    best_score=-np.inf
    for kernel in ['linear', 'rbf', 'poly']:
        gammas= [1] if kernel=='linear' else 10.0**np.arange(-3,0)
        degree= 2 if kernel=='poly' else None
        for gamma in  gammas:
            es=RankSVM(kernel=kernel, gamma=gamma, degree=degree)
            es.fit(np.array(training_set),pairwise)
            sc=es.score(np.array(training_set),pairwise)
            if sc>best_score:
                best_score=sc
                learner=es
    return learner
    
