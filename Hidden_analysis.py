import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
def formatting(string_numpy):
    """formatting : Conversion of String List to List
    
    Args:
        string_numpy (str)
    Returns:
        l (list): list of values
    """
    
    return [np.around(np.asarray(eval(re.sub(' +',',',x))),2) for x in string_numpy.values]



'''
Drawing the graphs
'''
for M in [4,10,20]:
    for Fixed in [False,True]:
        # M=4
        # Fixed=False
        suffix="Fixed" if Fixed else "Reduction"
        suffix=suffix+'-rMNK-{}'.format(M)
        #read experimental results into pandas data framework
        # File = open(r"Latex_image_input.txt","a+")
        filename ="Results"
        results_df = pd.read_csv(filename + ".csv")
        
        '''###Rename
        '''
        results_df.loc[(results_df['Mode']==3) & (results_df['recursive']==1), 'Mode']='k-HDR'
        results_df.loc[(results_df['Mode']==4) & (results_df['recursive']==1), 'Mode']='V-HDR'
        results_df.replace({"Mode": {1:"Golden", 2:"Only learning", 3:"k-HD", 4:"V-HD"}}, inplace=True)
        results_df.replace({"eq": {'eq10':"UF1", 'eq11':"UF2", 'eq12':"UF3", 'tch':'Tchebychef', 'quad':'Quadratic'}}, inplace=True)
        ##Important: Fixed-sized (has an space after "Only learning")*********
        results_df.loc[(results_df['Mode']=="Only learning") & (results_df['recursive']==0), 'Mode']="Only learning "
        
        '''###Filtering
        '''
        results_df=results_df[results_df['problemID']=='rMNK']
        results_df['K']=pd.to_numeric(results_df['K'], downcast='integer')
        results_df['r']=pd.to_numeric(results_df['r'], downcast='integer')
        
        if Fixed:
            ############Fixed-sized (has an space after "Only learning ")
            results_df=results_df[(results_df['Mode']!='Only learning')]
            results_df=results_df[(results_df['Mode']!='V-HD') & (results_df['Mode']!='V-HDR')]
        else:
            #############Variable number of objectives (NO space after "Only learning")
            results_df=results_df[(results_df['Mode']!='Only learning ')]
            results_df=results_df[(results_df['Mode']!='k-HD') & (results_df['Mode']!='k-HDR')] 
        
        
        '''
        table of results
        '''
        results_means=pd.pivot_table(results_df[results_df.NoOfInteractions.isin([6])], values='vf', index=['M','K'],
                                          columns=['eq','Mode'], aggfunc=np.mean, fill_value=0)
        print(results_means.to_latex(float_format="%.2f"))
        results_df=results_df[(results_df['NoOfInteractions']==6) & (results_df['M']==M)]
        '''###Drawing the Plots
        '''
        #Pivot table for calculating the average of vf values for runs in mode 3
        
        results_df['K']=pd.to_numeric(results_df['K'], downcast="integer")
        results_df['r']=pd.to_numeric(results_df['r'])
        
        probs=results_df['problemID'].unique()
        # modes=np.asarray(["Golden", "Only learning", "Learning + Detection", "V-HD", 'HDR', 'V-HDR'])# 
        # vf_means = pd.pivot_table(results_df, values='vf', index=['eq', 'K', 'Mode'],
        #                       columns=[ 'r'], aggfunc=np.mean, fill_value=0)
        # print(vf_means.to_latex())
        
        
        
        results_df['Mode'].unique()
        # algos=results_df['algo'].unique()
        interactions=results_df['NoOfInteractions'].unique()
        eqs=np.sort(results_df['eq'].unique())
        Ks=results_df['K'].unique()
        cors=results_df['r'].unique()
            
            
        # #Figures
        ncol=len(eqs)
        nrow=len(Ks)
        fig, axs = plt.subplots(nrow,ncol ,sharex=True, sharey=False,
                                squeeze=False, gridspec_kw={'hspace': 0.1},figsize=(18,12),
                                constrained_layout=True)
        
        palette=[ "C1", "C3","C0", "C4", "C6", "C7"][:len(results_df['Mode'].unique())]
        # graphs=nrow*ncol
        # for i in range(graphs-1,len(Ks)-1,-1):
        #     fig.delaxes(axs.flatten()[i])
        for eq in range(len(eqs)):
            for k in range(len(Ks)):
                data=results_df[(results_df['K']==Ks[k]) & (results_df['eq']==eqs[eq])]
                sns.set_style("white")
                #calculation of subfig indexes
                r=k#math.floor(k/ncol)
                c=eq#k%ncol
                sns.boxplot(data=data, x='r', y="vf", hue="Mode",
                                      ax=axs[r][c], fliersize=0, showfliers = False,
                                      palette='Greys')#palette
                sns.stripplot(data=data, x='r', y="vf", hue="Mode",
                                   palette='Greys', jitter=True,
                                   dodge=True,linewidth=1,edgecolor='gray',ax=axs[r][c])
                axs[r][c].get_legend().remove()
                axs[r][c].set(ylabel=None) 
                # axs[r][c].set(yticklabels=[])
                axs[r][c].set_xlabel('')
                # bw=hbw[(hbw['prob_id']=='rMNK') & (hbw['eq']==eqs[eq])]
                # best=bw['best'].iloc[0]
                # worst=bw['worst'].iloc[0]
                # axs[r][c].axhline(best,ls='--', color='blue')
                # axs[r][c].axhline(worst,ls='--', color='red')
                # axs[r][c].annotate(r"$K= {}, Eq{}$".format(int(Ks[k]), int(eqs[eq][-2:])-9),xy=(0.0,0), xycoords='axes fraction', xytext=(1.05, 0.3), textcoords='axes fraction', rotation='vertical')
                axs[r][c].annotate(r"{}".format(eqs[eq]),xy=(0.0,0), xycoords='axes fraction', xytext=(.4, 1.02), textcoords='axes fraction',fontsize=14)
                axs[r][c].annotate(r"K= {}, m={}".format(int(Ks[k]), M),xy=(0.0,0), xycoords='axes fraction', xytext=(1.05, 0.4), textcoords='axes fraction', rotation='vertical',fontsize=14)
                axs[r][c].margins(y=0)
                # axs[r][c].set(ylim=(data['vf'].values.min(), data['vf'].values.max()))
        # plt.tight_layout()
        handles, labels = axs[0][0].get_legend_handles_labels()
        #renaming labels for legend
        new_labels=[]
        for label in labels:
            if label == "Learning + Detection":
                new_label=r'$k-HD$'
            elif label=="V-HD":
                new_label=r'$\tau-HD$'
            elif label=="HDR":
                new_label=r'$k-HDR$'
            elif label=="V-HDR":
                new_label=r'$\tau-HDR$'
            else:
                new_label=label
            new_labels.append(new_label)
        labels=new_labels
        # l = plt.legend(handles[0:2], labels[0:2], borderaxespad=0., ncol=2), bbox_to_anchor=(0.25, 7),mode="expand"
        fig.legend(handles[:len(results_df['Mode'].unique())], labels[:len(results_df['Mode'].unique())] , ncol=len(results_df['Mode'].unique()), loc='upper center',borderaxespad=0, bbox_to_anchor=(0.55, 1.045,0,.01))#modes[0:len(results_df['Mode'].unique())]
        # fig.text(0.5, 1.01, r"Performance of the algorithm in different modes""\n"  r"problem $\rho$-MNK  for various values of $\rho$ and $K$" , ha='center', fontsize=14)
        fig.text(0.5, -.02, r'$\rho$', ha='center',fontsize=14)
        fig.text(-0.02, 0.5, 'Utility value', va='center', rotation='vertical',fontsize=14)
        plt.savefig("Utility"+suffix+".jpg", bbox_inches='tight', dpi=400)
        plt.show()

'''
Time pivot
'''

time_means=pd.pivot_table(results_df, values='time',
                                  columns=['Mode'], aggfunc=np.mean, fill_value=0)
print(time_means.to_latex(float_format="%.2f"))

'''
Trend Analysis with regard to identified objs
'''
df= pd.read_csv("identified_objs.csv")
df.dropna(how='all', axis=1, inplace=True)
df2=pd.read_csv("hidden_jobs.csv", names=['idi', 'p','problemID', 'Mode1', 'NoOfInteractions','run','eq','M', 'r', 'K', 'tau', 'recursive'], index_col=False)
df=pd.merge(df,   df2,  on ='idi',   how ='left', suffixes=["_y", ""])
# df['K']=pd.to_numeric(results_df['K'], downcast="integer")
df=df[(df['Interactions']==6) & (df['M']==M)]
df=df[(df['Mode']==3) | (df['Mode']==4)]
df.replace({"Mode": {1:"Golden", 2:"Only learning", 3:"Learning + Detection", 4:"V-HD"}}, inplace=True)
df.replace({"eq": {'eq10':1, 'eq11':2, 'eq12':3}}, inplace=True)
eqs=df['eq'].unique()
Ks=df['K'].unique()
cors=df['r'].unique()
# df['f']=formatting(df['f'])
nrow=len(cors)*len(Ks)
ncol=len(probs)*len(eqs)
fig, axs = plt.subplots(nrow,ncol ,sharex=True, sharey=False,
                                squeeze=False, gridspec_kw={'hspace': 0.1},figsize=(10,30))#,constrained_layout=True
# ncol=len(eqs)
# nrow=len(probs)
for cor in range(len(cors)):
    for k in range(len(Ks)):
        data=df[(df['r']==cors[cor]) & (df['K']==Ks[k])]
        #
        # graphs=math.ceil(len(probs)/ncol)*ncol
        # for i in range(graphs-1,len(probs)-1,-1):
        #     fig.delaxes(axs.flatten()[i])
        vf_means=pd.pivot_table(data, values='vf', index=['problemID', 'eq'],
                                  columns=['interaction'], aggfunc=np.mean, fill_value=0)
        for prob in range(len(probs)) :
           for eq in range(len(eqs)):
        
                # plt.figure(figsize=(6,3))
                # sns.set_style("white")
                #calculation of subfig indexes
                r=cor*len(Ks)+k#math.floor(prob/ncol)
                c=prob*len(eqs)+eq#prob%ncol
                axs[r][c].plot(vf_means.loc[probs[prob]].loc[eqs[eq]].index.values,vf_means.loc[probs[prob]].loc[eqs[eq]].values )
                
        
                axs[r][c].set(yticklabels=[])
                # bw=hbw[(hbw['prob_id']=='rMNK') & (hbw['eq']==eqs[eq])]
                # best=bw['best'].iloc[0]
                # worst=bw['worst'].iloc[0]
                # axs[r][c].axhline(best,ls='--', color='blue')
                # axs[r][c].axhline(worst,ls='--', color='red')
                # axs[r][c].set(xticklabels=[])
                # axs[r][c].set_xlabel('')
                axs[r][c].margins(y=0)
                axs[r][c].annotate(r'$\rho$'"={}, K={}\nEq {}".format(cors[cor], Ks[k],eqs[eq] ),xy=(0.0,0), xycoords='axes fraction', xytext=(1.01, 0.1), textcoords='axes fraction', rotation='vertical')

fig.text(0.5, 0.111, 'Number of interactions', ha='center',fontsize=14)
fig.text(0.1, 0.5, 'Utility value', va='center', rotation='vertical',fontsize=14)
pdf_filename = "Hidden-rMNK-trend"+suffix+".pdf"
plt.savefig(pdf_filename, bbox_inches='tight')
plt.show()

##HeatMap (First time finding the true set of OBJs)
df['sc1']=0
df['sc2']=0
df.loc[df.dv1.isin([0,1]), 'sc1'] = 1
df.loc[df.dv2.isin([0,1]), 'sc2'] = 1
df['sc']=df['sc1']+df['sc2']

for i in df.idi.unique():
    qqq=df[df.idi==i]['sc'].values
    qqq=np.maximum.accumulate(qqq)-1
    df.loc[df.idi==i,'sc']=qqq
    
vf_means = pd.pivot_table(df, values='sc', index=['Problem','K','r','eq'],
                          columns=['interaction'], aggfunc='mean', fill_value=0)

print(vf_means.to_latex(float_format="%.2f"))

'''
Distance calculations (from the preferred point)
'''
'''
filename="identified_objs"+'.csv'
df= pd.read_csv(filename)
df2=pd.read_csv("hidden_jobs.csv", names=['idi', 'problemID', 'Mode1', 'NoOfInteractions','run','eq', 'M','r', 'K'], index_col=False)
# df=df.join(df2.set_index('idi'), on='idi', validate)
df=pd.merge(df,   df2,  on ='idi',   how ='left')#, validate='many_to_one'
df=df[(df['Mode']==3)&(df['Interactions']==6) & (df['M']==M) & (df['problemID']=='rMNK')]
# df.replace({"Mode": {1:"Golden", 2:"Only learning", 3:"Learning + Detection"}}, inplace=True)
df.replace({"eq": {'eq10':1, 'eq11':2, 'eq12':3}}, inplace=True)
hbw.replace({"eq": {'eq10':1, 'eq11':2, 'eq12':3}}, inplace=True)
hbw=hbw[((hbw['M']==M) & (hbw['prob_id']=='rMNK'))]
hbw["r"] = pd.to_numeric(hbw["r"], downcast="float")
hbw["k"] = pd.to_numeric(hbw["k"], downcast="float")
# hbw['best_sol']=formatting(hbw['best_sol'])
df['f']=formatting(df['f'])

for eq in range(len(eqs)):
    for cor in range(len(cors)):
        for k in range(len(Ks)):
            opt=hbw[(hbw['eq']==eqs[eq]) & (hbw['r']==cors[cor]) & (hbw['k']==Ks[k])]['best_sol'].iloc[0]
            for i in df.loc[(df['eq']==eqs[eq]) & (df['r']==cors[cor]) & (df['K']==Ks[k])].index:
                point=df.loc[i,'f']
                df.loc[i,'dis']=np.linalg.norm(opt-point)
       
       
       
# df.replace({"eq": {'eq10':1, 'eq11':2, 'eq12':3}}, inplace=True)       
vf_means = pd.pivot_table(df, values='dis', index=['eq',  'r'],
                          columns=['interaction'], aggfunc='mean', fill_value=0)#'K','Problem'
print(vf_means.to_latex(float_format="%.2f"))

'''


# fig, ax = plt.subplots()
# im = ax.imshow(vf_means, cmap="gray_r")
# x_label_list = list(range(1,11))

# ax.set_xticks(list(range(10)))

# ax.set_xticklabels(x_label_list)

# y_label_list = list(range(1,11))

# ax.set_yticks(list(range(10)))

# ax.set_yticklabels(y_label_list)
# fig.text(0.5, 1.1, r'Trend of VF: $\rho$={} and $K$={}'.format(cors[cor],Ks[k] ) , ha='center', fontsize=14)    
# sns.stripplot(data=results_df, x='r', y="vf",
#                       hue="Mode", palette="colorblind", jitter=True,
#                       split=True,linewidth=1,edgecolor='gray')

# Creating plots for different rhos and Ks. Only interaction =6
# results_df=results_df[ (results_df['Mode']==3)]#(results_df['NoOfInteractions']==6) &
# #Pivot table for calculating the average of vf values for runs
# ind='r'
# col='NoOfInteractions'

# vf_means = pd.pivot_table(results_df, values='vf', index=[ind],
#                       columns=[ col], aggfunc=np.mean, fill_value=0)
# # fig, axs = plt.subplots(nrow,ncol ,sharex=True, sharey=False,
# #                         squeeze=False, gridspec_kw={'hspace': 0.1},figsize=(6,4),
# #                         constrained_layout=True)
# lines = ["-","--","-.",":"]
# linecycler = cycle(lines)
# for k in vf_means.index.values:
#     plt.plot(vf_means.columns.values, vf_means.loc[k].values,  next(linecycler), label= ind+' ={}'.format(k))
# plt.legend()
# plt.show()
    
    
    # '''
    # ===================================================================
    # Plotting Pf, worst and best solutions
    # ===================================================================
    # '''

    # if fDim==2:
    #     problem = pg.problem(pg.dtlz(prob_id = int(data['problemID'].unique()[0][-1]), dim =Dim, fdim = fDim, alpha = 100))
        
    #     f_name="{}-{}_PF.npy".format(problem.get_name(), problem.get_nobj())
    #     pf = np.load(f_name)
    #     u=vf.stewart_value_function(w=np.full(fDim,1./fDim), tau=np.full(fDim,tau), alpha=np.full(fDim,alpha), beta=np.full(fDim,beta), lambd=np.full(fDim,lambd), delta=np.full(fDim,0))
    #     dm=MachineDM.machineDM(problem, u, 3, 0, 0, 0, np.full(fDim,0.5),scaling=True, ideal= ideal, nadir=nadir)
    #     u_min=table_of_probs.loc[probs[prob],'optimum utility']
    #     u_max=table_of_probs.loc[probs[prob],'Worst Utility']
    #     pf_u = np.apply_along_axis(dm.value, axis=1, arr = pf)
    #     u_argmin = np.argmin(pf_u)
    #     u_argmax = np.argmax(pf_u)
        
    #     #contours
        
    #     f=parse_array(data['f'])
    #     x=np.linspace(ideal[0],max(nadir[0],f.max(axis=0)[0]),200)
    #     y=np.linspace(ideal[1],max(nadir[1],f.max(axis=0)[1]),200)
    #     X,Y=grid = np.meshgrid(x, y)
    #     Z=np.stack((X,Y), axis=-1)
    #     Z=np.apply_along_axis(dm.value,axis=-1,arr=Z)
    #     # scaler = MinMaxScaler()
    #     # Z=scaler.fit_transform(Z)
        
        
        
        
    #     # plotting the found points in different conditions (non-idealities)
    #     for c in range(len(data['NoOfInteractions'].unique())):
    #         # plt.clf()
    #         #plotting Pf, worst and best
    #         plt.contourf(X, Y, Z, levels=40,color='RdBu')
    #         plt.colorbar()
    #         plt.scatter(pf[:,0], pf[:,1],s=1,color='k',edgecolors='w')

    #         for algo in algos:
    #             f=data[(data['NoOfInteractions']==data['NoOfInteractions'].unique()[c]) & (data['algo']==algo)]['f']
    #             f=parse_array(f)
    #             plt.scatter(f[:,0],f[:,1], label=" {}".format(algo ),color= 'y' if algo=='iTDEA' else 'b')#, facecolors='none', edgecolors='w'
    #         plt.scatter(pf[u_argmin][0], pf[u_argmin][1],s=250,marker='+', color='g', label='Most preferred')
    #         plt.scatter(pf[u_argmax][0], pf[u_argmax][1],s=150,marker='x', color='red', label= 'Worst PF solution')
            
    #         plt.title("prob {}, {} with {} objectives, {}: {}".format(probs[prob], data['problemID'].unique()[0], fDim, 'Intreactions',data['NoOfInteractions'].unique()[c]))
    #         plt.legend(bbox_to_anchor=(1.3, 1), loc='upper left')
    #         pdf_filename = "Contour- prob {}- Problem{}-{} objectives {} level {}".format(probs[prob], data['problemID'].unique(), fDim, 'Intractions',c+1)+".pdf"
    #         # plt.savefig(pdf_filename, bbox_inches='tight')
    #         plt.show()

# File.close()    


