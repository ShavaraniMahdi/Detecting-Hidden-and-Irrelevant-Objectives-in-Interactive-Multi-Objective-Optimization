# =============================================================================
# Experiments as outlined in:
#     Battiti, R., & Passerini, A. (2010). Brainâ€“computer evolutionary multiobjective optimization: a genetic 
#     algorithm adapting to the decision maker. IEEE Transactions on Evolutionary Computation, 14(5), 671-687.
# Some notes from the paper section "VI. Experimental Evaluation":
#     -We chose a population size of
#     100, 500 generations, probability of crossover equal to 1 and
#     probability of mutation equal to the inverse of the number of
#     decision variables.The number of initial generations (gen1)
#     was set to 200, while the number of generations between
#     two training iterations (geni) was set to 20 (15 iterations).
#     A perfect fit (thres = 1) was required in order to prematurely stop training.
#     The number of training iterations and examples per iteration
#     were varied in the experiments as detailed later on in the
#     section.Concerning the learning algorithm, we fixed the C
#     regularization parameter in (8) to 100.
#     For test problem DTLZ1, we
#     slightly restricted the feasible set for decision variables with
#     respect to the original formulation (from [0, 1] to [0.25, 0.75]) ...##M: If we could not change the bounds with a problem method in pygmo, we can interploate all the population values pop/2+0.25
#     The first set of experiments aims at showing the effectiveness
#     of the method in early focusing on the correct search area
#     with very few queries to the DM, for different test problems
#     and number of objectives, in the setting of linear utility
#     functions. For each problem class, we generated a number
#     of test instances by varying the number of objectives m from
#     2 to 10 and by setting the size of the input decision vector to
#     n = 100 for 0/1 knapsacks, n = 2m for DTLZ1 and DTLZ6,
#     and n = 10m, as suggested in [46], for DTLZ7. A total of 36
#     test instances was thus considered. Furthermore, we generated
#     linear utility functions by randomly choosing weights in the
#     range (0, 1].
#         Each
#     graph reports three learning curves for an increasing number
#     of training examples per iteration (exa), with one, two, and
#     three iterations (maxit), respectively. Results are the medians
#     over 100 runs with different random speeds for the search of
#     the EA.
# =============================================================================
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import sys,time
import pygmo as pg
import bcemoaObjDet
import MachineDM
import value_function
import problems
# 2403,	rMNK,	2,	6,	1,	eq10,	10	0	15
#2403,6,3,6,12,'eq10'#
# idi, p,  prob_id, mode, interactions, run, eq, M,r,k , thre, recursive= [22526,'DTLZ',7,2,3,38,'eq10',20,0,0,0,0]
thre=float(sys.argv[11])
threshold=None if thre==0 else thre
recursive=int(sys.argv[12])
idi=None
archive=True
idi, p,  prob_id, mode, interactions, run, eq =[int(sys.argv[1]), sys.argv[2], sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]), sys.argv[7]]
M=int(sys.argv[8])
#test problem configs
if prob_id=='rMNK':
    r,K=[float(sys.argv[9]), int(sys.argv[10])]
    active=[0,1]
else:
    active=[0,3]
    r,K=['-','-']
    prob_id=int(prob_id)
if mode!=4:
    threshold=None
# Training samples in each interaction
exa=5

# Results
col_names = columns=['idi','problemID', 'Mode', 'NoOfInteractions', 'exa', 'Run', 'approximation_error', 'vf','f', 'x', 'r', 'M', 'N', 'K', 'eq', 'time', 'threshold', 'recursive']
results_df = pd.DataFrame(columns=col_names)
# vf_names = ["eq10"]

vf_dict = {
    # Utility equations in BC-EMOA paper
    'eq10': ('0.28*x[0]**2   + 0.29*x[0]*x[1] + 0.38*x[1]**2 + 0.05*x[0]', [1,3]),
    'eq11': ('0.05*x[0]*x[1] + 0.6*x[0]**2 + 0.38*x[1] + 0.23*x[0]', [1,3]),
    'eq12': ('0.44*x[0]**2 + 0.33*x[0] + 0.09*x[0]*x[1] + 0.14*x[1]**2', [1,3]),
    # 'eq14': ('0.44*x[2]**2 + 0.33*x[2] + 0.09*x[2]*x[3]  +0.14*x[3]**2', [1,3])
};

vf_dict_DTLZ127 = {
    # Utility equations in BC-EMOA paper
    'eq10': ('0.28*x[0]**2   + 0.29*x[0]*x[3] + 0.38*x[3]**2 + 0.05*x[0]', [1,3]),
    'eq11': ('0.05*x[0]*x[3] + 0.6*x[0]**2 + 0.38*x[3] + 0.23*x[0]', [1,3]),
    'eq12': ('0.44*x[0]**2 + 0.33*x[0] + 0.09*x[0]*x[3] + 0.14*x[3]**2', [1,3])
    # ,
    # 'eq14': ('0.44*x[2]**2 + 0.33*x[2] + 0.09*x[2]*x[3]  +0.14*x[3]**2', [1,3])
};
    
def get_vf(prob, vf_name):
    if vf_name=='tch':
        weights=np.zeros(M)
        weights[active]=0.5
        ideal=np.zeros(M)
        if prob_id==1:
            weights[active]=np.array([0.97, 0.03])
            # ideal[active]=np.array([0.15912147, 0.30837088])
        
        if prob_id==2:
            weights[active]=np.array([0.79835788, 0.20164212])
            # ideal[active]=np.array([0.03453165, 0.08893429])
    
        if prob_id==7:
            weights[active]=np.array([0.80588792, 0.19411208])
            # ideal[active]=np.array([0.52105879, 3.35404389])
        return value_function.tchebycheff_value_function(weights, ideal)
            
    
        
    if vf_name=='quad':
        weights=np.zeros(M)
        weights[active]=0.5
        ideal=np.zeros(M)
        if prob_id==1:
            weights[active]=np.array([0.91972123, 0.08027877])
            # ideal[active]=np.array([0.33963493, 0.25326929])
        
        
        if prob_id==2:
            weights[active]=np.array([0.98232557, 0.01767443])
            # ideal[active]=np.array([0.03181192, 0.50100643])
    
        if prob_id==7:
            weights[active]=np.array([0.78805194, 0.21194806])
            # ideal[active]=np.array([0.48455603, 3.46706611])
        return value_function.quadratic_value_function(weights, ideal)

    else:
        if prob_id=='rMNK':
            vf = vf_dict[vf_name][0]
        else:
            vf = vf_dict_DTLZ127[vf_name][0]
        return value_function.polynomial(vf)
        
#value Function

# def pop_original_fitness(prob, pop):
#     fitness = []
#     for ind in range(len(pop)):
#         fitness.append(prob.original_fitness(pop.get_x()[ind]))
#     return np.asarray(fitness)

def get_mdm(prob, vf_name):
    vf = get_vf(prob, vf_name)
    #machine dm
    # mode: 1: without interaction (gold standard), 2: with interaction and learning, Detection off, without biases, 3: with interactions and learning but wiht detection of objs
    #obj modification factors: gamma: dependency among obj values, Sigma: noise control, delta: shift in the reference levels, q:number of objs not modelled
    gamma=0
    sigma=0
    #delta=0
    q=0
    tau = np.full(prob.prob.get_nobj(),[0.5])
    return MachineDM.machineDM(prob.prob, vf, mode, gamma, sigma, q, tau)

def run_bcemoa(pop, mode, interactions, run, prob, exa, vf_name):
    #(prob,pref,mode, gamma, sigma, delta, q,seed = pg.random_device.next() ):
    detection=1 if (mode==3 or mode==4) else 0
    gen1=50 if (mode==4) else 50
    mdm = get_mdm(prob, vf_name)
    # The algorithm (a self-adaptive form of Differential Evolution (sade - jDE variant)
    m_m = 1 / len(pop.get_x()[0])
    algo = pg.algorithm(bcemoaObjDet.bcemoa(
        mdm, gen1 = gen1, cr = 0.99, eta_c = 10., m = m_m, eta_m = 50,
        seed=run**5,
        geni=30, interactions=interactions, sampleSize=exa, detection=detection, verbose=0, idi=idi, archive=archive, threshold=threshold, recursive=recursive))#
    # The actual optimization process
    pop = algo.evolve(pop)
    return pop



# The problem
#modeleled/dummy vector is a binary vector that represents the modeled objectives in a problem, if it is not passed
#to the problem it means all the objectives are modele
vf_name=eq
if prob_id==7:
    N=M+19
    problem = pg.problem(pg.dtlz(prob_id = prob_id ,fdim = M ,dim = N,  alpha = 100))
elif prob_id==1:
    N=M+4
    pr = pg.dtlz(prob_id = prob_id ,fdim = M ,dim = N,  alpha = 100)
    problem =problems.bounded_prob(pr,0.25,0.75)
elif prob_id==2:
    N=M+9
    pr = pg.dtlz(prob_id = prob_id ,fdim = M ,dim = N,  alpha = 100)
    problem =problems.bounded_dtlz2(pr,0,1)
    #problem = pg.problem(pg.dtlz(prob_id = 6, dim = N, fdim = M, alpha = 100))
    #problem = pg.problem(pg.dtlz(prob_id = 6, dim = N, fdim = M, alpha = 100))
elif prob_id=='MNK':
    N=10 if M==4 else 20
    problem = pg.problem(problems.MNK(M,N,K))
elif prob_id=='rMNK':
    N=(round(M/10)+1)*10
    # N=10 if M==4 else 20
    problem = pg.problem(problems.rMNK(r,M,N,K))

if (mode==1):
    dummyVector=np.asarray(active)
elif (mode==2 and threshold==None) or mode==3:
    dummyVector=np.asarray([0,2])
else:
    dummyVector=np.arange(problem.get_nobj()) 
dummy_prob = problems.dummy(problem, dummyVector)
prob=dummy_prob
np.random.seed(run**4)
# The initial population
pop = pg.population(prob, size = 200, seed=run**4)
# fitness = pop_original_fitness(dummy_prob, pop)
# scaler.partial_fit(fitness)
vf=get_vf(prob, vf_name)
start1 = time.time()
pop = run_bcemoa(pop, mode, interactions, run, prob, exa, vf_name)
end1=time.time()
fitness = dummy_prob.original_fitness(pop.get_x()[0])

result_df = pd.DataFrame([[fitness]], columns=['f'])
result_df['Mode'] = mode
result_df['NoOfInteractions'] = interactions
result_df['Run'] = run
result_df['problemID'] = prob_id
result_df['x'] = [np.asarray(pop.get_x()[0])]
result_df['exa'] = exa
result_df['vf']=vf.value(fitness)
result_df['identified1', 'identified2']=[pop.problem.extract(problems.dummy).get_dv()]
result_df['eq']=eq
result_df['idi']=idi
result_df['r'] = r
result_df['M'] = M
result_df['N'] = N
result_df['K'] = K
result_df['prob'] = p
result_df['recursive'] = recursive
result_df['threshold'] = threshold
result_df['time'] = end1-start1
results_df = results_df.append(result_df, ignore_index=True,sort=False)
# print(results_df)


    # prob = problems.dummy(problem, dummyVector)
    # mdm = get_mdm(prob, vf_name)
    # golden_pop = run_bcemoa(pg.population(prob, size = 300), 1, 1, 12345, prob, 5, vf_name)
    # golden_standard=mdm.value(golden_pop.get_f()[0])
    # for index, row in results_df[results_df['problemID'] == prob_id].iterrows():
    #     fitness = row[['f1', 'f2']]#['f1', 'f2','f3', 'f4']
    #     results_df.loc[index, 'vf'] = mdm.value(fitness) #fitness[None,:]  #mdm.value(scaler.transform(fitness[None,:]))
    #     results_df.loc[index, 'approximation_error'] = 100*(mdm.value(fitness)-golden_standard)/golden_standard
    # results_df['vf_scaled'] = scaler.fit_transform(results_df['vf'].values.reshape(-1, 1))


filename = "experiments-{}-".format(idi) + ".csv"
results_df.to_csv(filename, index=False)
#today = date.today().strftime("%Y-%m-%d")
# time=str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
# filename = "experiments-{}-".format(vf_name) + time + ".csv"
# results_df.to_csv(filename, index=None)
# print("Saved results to " + filename + "\n")

