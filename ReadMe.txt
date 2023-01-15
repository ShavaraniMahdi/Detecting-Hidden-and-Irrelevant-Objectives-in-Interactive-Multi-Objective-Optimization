Run command:

python bcemoaObjDet_experiments.py -idi -problem  -prob_id -mode -interactions -run -eq -M -r -k -thre -recursive

idi: Id of the test.
problem: problem name{DTLZ or rMNK}
problem_id: DTLZ id number (1-7), for rMNK (-)
mode: 1 Golden, 2 Only learning, 2 Learning + detection 
interactions: number of interactions
run: run number (used also to seed the experiment)
eq: the equation number of the utility value
M: number of objectives
r: rho value for rMNK problems
k: K value of rMNK problems
thre: threshold used in variable number of objectives, None if number of objectives is fixed
recursive: if recursive feature selection should be use


For analysis of the results of experiments on dtlz and rmnk prblem, respetctively run "Hidden_analysis167.py" and "Hidden_analysis.py" script. it will read the results and generate the graphs.