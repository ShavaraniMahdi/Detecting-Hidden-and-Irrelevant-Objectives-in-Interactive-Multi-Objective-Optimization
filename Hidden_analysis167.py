import sys
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pygmo as pg
import pandas as pd
import seaborn as sns
import math
from itertools import cycle
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler
from ea_operators import problem_ideal_nadir
import value_function as vf
import MachineDM
from matplotlib import ticker

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def formatting(string_numpy):
    """formatting : Conversion of String List to List
    
    Args:
        string_numpy (str)
    Returns:
        l (list): list of values
    Delets spaces at beginin and end and replaces spaces in the middle with commas,
    Then converts to array
    """
    return [np.around(np.asarray(eval(re.sub(' +',',',re.sub(r'\[\s+', '[', re.sub(r'\s+\]', ']', x))))),2) for x in string_numpy.values]



'''
Drawing the graphs
'''

for M in [4,10,20]:
    for Fixed in [False,True]:
        # M=4
        # Fixed=True
        suffix="Fixed" if Fixed else "Reduction"
        suffix=suffix+'-127-{}'.format(M)
        
        #read experimental results into pandas data framework
        # File = open(r"Latex_image_input.txt","a+")
        filename ="Results"
        results_df = pd.read_csv(filename + ".csv")
        # results_df.replace({"problemID": {'7':7}}, inplace=True)
        
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
        results_df=results_df[results_df['problemID']!='rMNK']
        
        
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
        results_means=pd.pivot_table(results_df[results_df.NoOfInteractions.isin([6])], values='vf', index=['M','problemID'],
                                          columns=['eq','Mode'], aggfunc=np.mean, fill_value=0)
        print(results_means.to_latex(float_format="%.2f"))
        
        results_df=results_df[results_df.NoOfInteractions.isin([1,3,6]) & (results_df['M']==M)]
        
        
        probs=results_df['problemID'].unique()
        modes=results_df['Mode'].unique()
        interactions=results_df['NoOfInteractions'].unique()
        eqs=sorted(results_df['eq'].unique())
        
        sns.color_palette('Greys')
        ncol=len(eqs)
        nrow=len(probs)
        fig, axs = plt.subplots(nrow,ncol ,sharex=True, sharey=False,
                                squeeze=False, gridspec_kw={'hspace': 0.1},figsize=(15,8),
                                constrained_layout=True)#
        # graphs=math.ceil(len(probs)/ncol)*ncol
        # for i in range(graphs-1,len(probs)-1,-1):
        #     fig.delaxes(axs.flatten()[i])
        for prob in range(len(probs)) :
            for eq in range(len(eqs)):
               
                # plotting boxplots for comparison of algorithms performance in mode 3 (wiht biases if applied)
                data = results_df[(results_df["problemID"] == probs[prob]) & (results_df["eq"] == eqs[eq])]
            
                # For scaling the outliers uncomment the following lines
                # data['vf'] = data['vf'].apply(lambda x: x if x <= worst*1.02 else worst*1.02)
                # plt.figure(figsize=(6,3))
                sns.set_style("white")
                #calculation of subfig indexes
                r=prob#math.floor(prob/ncol)
                c=eq#prob%ncol
                   
                            # palette="Set2", 
                sns.boxplot(data=data, x='NoOfInteractions', y="vf",
                                  hue="Mode",fliersize=0,palette='Greys',
                                  ax=axs[r][c],showfliers=False)#, =palette
                
                ylims=axs[r][c].get_ylim()
                sns.stripplot(data=data, x='NoOfInteractions', y="vf",
                                  hue="Mode", jitter=True,palette='Greys',
                                  dodge=True,linewidth=1,edgecolor='gray',ax=axs[r][c])#, palette=palette
                axs[r][c].get_legend().remove()
                axs[r][c].set(ylabel=None) 
                # axs[r][c].set(yticklabels=[])
                axs[r][c].set_xlabel('')
                
                # bw=hbw[(hbw['prob_id']==str(probs[prob])) & (hbw['eq']==eqs[eq]) & (hbw['M']==M)]
                # best=bw['best'].iloc[0]
                # worst=bw['worst'].iloc[0]
                # axs[r][c].axhline(best,ls='--', color='blue')
                # axs[r][c].axhline(worst,ls='--', color='red')
                axs[r][c].margins(y=0)
                axs[r][c].set(ylim=ylims)
                # ticker.FormatStrFormatter('%.2e')
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True) 
                # formatter.set_powerlimits((-1,1))
                # formatter.scilimits=2
                formatter.useOffset=False
                axs[r][c].yaxis.set_major_formatter(formatter) 
                # _,p_value=wilcoxon(results_df[(results_df["problemID"]==probs[prob])&(results_df["Mode"]=="Only learning")]["vf"].values,
                #                 results_df[(results_df["problemID"]==probs[prob])&(results_df["Mode"]=="Learning + Detection")]["vf"].values,
                #                 zero_method="zsplit")
                # axs[r][c].annotate("DTLZ{}, Eq {}\n p-value={:.3f}".format(probs[prob],int(eqs[eq][-2:])-9, p_value),xy=(0.0,0), xycoords='axes fraction', xytext=(1.05, 0.01), textcoords='axes fraction', rotation='vertical')
                axs[r][c].annotate("{}".format(eqs[eq]),xy=(0.0,0), xycoords='axes fraction', xytext=(.4, 1.03), textcoords='axes fraction', fontsize=14)#
                axs[r][c].annotate("DTLZ{}, m={}".format(probs[prob],M),xy=(0.0,0), xycoords='axes fraction', xytext=(1.05, 0.4), textcoords='axes fraction', rotation='vertical', fontsize=14)#
        
            
        # plt.tight_layout()
        handles, labels = axs[0][0].get_legend_handles_labels()
        labels=[w.replace('V-HD', r'$\tau-HD$') for w in labels]
        labels=[w.replace('V-HDR', r'$\tau-HDR$') for w in labels]
        # l = plt.legend(handles[0:2], labels[0:2], borderaxespad=0., ncol=2), bbox_to_anchor=(0.25, 7),mode="expand"
        fig.legend(handles,labels[0:len(modes)], ncol=len(modes), loc='upper center',borderaxespad=.0, bbox_to_anchor=(0.52, 1.06,0,.01))#
        # fig.text(0.5, 1.05, 'Comparison of the performance of the algorithm in different modes' , ha='center', fontsize=14)
        fig.text(0.5, -.02, 'Number of interactions', ha='center',fontsize=14)
        fig.text(-0.01, 0.5, 'Utility value', va='center', rotation='vertical',fontsize=14)
        plt.savefig("utility"+suffix+".pdf", bbox_inches='tight', dpi=400)
        plt.show()








'''
Time pivot
'''

# time_means=pd.pivot_table(results_df, values='time', index=['problemID'],
#                                   columns=['Mode'], aggfunc=np.mean, fill_value=0)
# print(time_means.to_latex(float_format="%.2f"))





'''
Trend Analysis with regard to identified objs
'''
M=4
filename="identified_objs"+'.csv'
df= pd.read_csv(filename)
df2=pd.read_csv("hidden_jobs.csv", names=['idi', 'prob', 'problemID', 'Mode', 'NoOfInteractions','run','eq', 'M','r', 'K', 'tau', 'recursive'], index_col=False)
# df=df.join(df2.set_index('idi'), on='idi', validate)
results_df=pd.merge(df,   df2,  on ='idi',   how ='left', suffixes=["", "_y"])
results_df.loc[(results_df['Mode']==3) & (results_df['recursive']==1), 'Mode']='k-HDR'
results_df.loc[(results_df['Mode']==4) & (results_df['recursive']==1), 'Mode']='V-HDR'
results_df.replace({"Mode": {1:"Golden", 2:"Only learning", 3:"Learning + Detection", 4:"V-HD"}}, inplace=True)
results_df=results_df[(results_df['Mode']!='V-HDR') & (results_df['Mode']!='k-HDR') & (results_df['Mode']!='Golden') & (results_df['Mode']!='Learning + Detection') & (results_df['eq']!='tch') & (results_df['eq']!='quad') & (results_df['M']==M) & (results_df['problemID']!='rMNK') & (results_df['Interactions']==6)]
# df.replace({"Mode": {1:"Golden", 2:"Only learning", 3:"Learning + Detection"}}, inplace=True)
results_df.replace({"eq": {'eq10':1, 'eq11':2, 'eq12':3}}, inplace=True)
probs=results_df['problemID'].unique()
eqs=results_df['eq'].unique()
ncol=len(eqs)
nrow=len(probs)
fig, axs = plt.subplots(nrow,ncol ,sharex=True, sharey=False,
                        squeeze=False, gridspec_kw={'hspace': 0.1},figsize=(18,12),
                        constrained_layout=True)


vf_means=pd.pivot_table(results_df, values='vf', index=['problemID', 'eq'],
                          columns=['interaction'], aggfunc=np.mean, fill_value=0)
vf_std=pd.pivot_table(results_df, values='vf', index=['problemID', 'eq'],
                          columns=['interaction'], aggfunc=np.std, fill_value=0)

for eq in range(len(eqs)):
    for prob in range(len(probs)):
        # plt.figure(figsize=(6,3))
        sns.set_style("white")
        #calculation of subfig indexes
        r=prob#math.floor(prob/ncol)
        c=eq#prob%ncol
        x=vf_means.loc[probs[prob]].loc[eqs[eq]].index.values
        y=vf_means.loc[probs[prob]].loc[eqs[eq]].values 
        y_std=vf_std.loc[probs[prob]].loc[eqs[eq]].values
        ci = .196 * y_std/np.sqrt(len(x))
        axs[r][c].plot(x,y)
        axs[r][c].fill_between(x, (y-ci), (y+ci), color='b', alpha=.07)

        # axs[r][c].set(yticklabels=[])
        axs[r][c].set_xlabel('')
        axs[r][c].margins(y=0)
        axs[r][c].annotate("DTLZ {}, UF{}".format(probs[prob],eqs[eq]),xy=(0.0,0), xycoords='axes fraction', xytext=(1.05, 0.3), textcoords='axes fraction', rotation='vertical', fontsize=20)
        
        # bw=hbw[(hbw['prob_id']==str(probs[prob])) & (hbw['eq']==eqs[eq])]
        # best=bw['best'].iloc[0]
        # worst=bw['worst'].iloc[0]
        # axs[r][c].axhline(best,ls='--', color='blue')
        # axs[r][c].axhline(worst,ls='--', color='red')

# plt.tight_layout()
pdf_filename = "DTLZ_trend_{}".format(M)+suffix+".jpg"
# handles, labels = axs[0][0].get_legend_handles_labels()
# # l = plt.legend(handles[0:2], labels[0:2], borderaxespad=0., ncol=2), bbox_to_anchor=(0.25, 7),mode="expand"
# fig.legend(handles,labels[0:3], ncol=3, loc='upper center',borderaxespad=3)
#fig.text(0.5, 1.05, 'Utility value of best solution after each interaction' , ha='center', fontsize=20)
fig.text(0.5, -.02, 'Number of interactions', ha='center',fontsize=20)
fig.text(-0.05, 0.5, 'Utility value', va='center', rotation='vertical',fontsize=20)
plt.savefig(pdf_filename, bbox_inches='tight', dpi=400)
plt.show()


##HeatMap (First time finding the true set of OBJs)
# df['sc1']=0
# df['sc2']=0
# df.loc[df.dv1.isin([0,1]), 'sc1'] = 1
# df.loc[df.dv2.isin([0,1]), 'sc2'] = 1
# df['sc']=df['sc1']+df['sc2']
# vf_means = pd.pivot_table(df, values='sc', index='Problem',
#                           columns=['interaction'], aggfunc='mean', fill_value=0)
# for i in df.idi.unique():
#     qqq=df[df.idi==i]['sc'].values
#     qqq=np.maximum.accumulate(qqq)-1
#     df.loc[df.idi==i,'sc']=qqq
# df.replace({"eq": {'eq10':1, 'eq11':2, 'eq12':3}}, inplace=True)
    
# vf_means = pd.pivot_table(df, values='sc', index=['Problem', 'eq'],
#                           columns=['interaction'], aggfunc='mean', fill_value=0)

# print(vf_means.to_latex(float_format="%.2f"))



'''
Distance calculations (from the preferred point)
'''

'''
df= pd.read_csv("identified_objs"+".csv")

df2=pd.read_csv("hidden_jobs.csv", names=['idi', 'problemID', 'Mode1', 'NoOfInteractions','run','eq', 'M'], index_col=False)
# df=df.join(df2.set_index('idi'), on='idi', validate)
df=pd.merge(df,   df2,  on ='idi',   how ='left')#, validate='many_to_one'
df=df[(df['Mode']==3)&(df['Interactions']==6) & (df['M']==M)]
df.replace({"Mode": {1:"Golden", 2:"Only learning", 3:"Learning + Detection"}}, inplace=True)
df.replace({"eq": {'eq10':1, 'eq11':2, 'eq12':3}}, inplace=True)
hbw.replace({"eq": {'eq10':1, 'eq11':2, 'eq12':3}}, inplace=True)
hbw=hbw[(hbw['M']==M)]
eqs=df['eq'].unique()
hbw['best_sol']=formatting(hbw['best_sol'])
df['f']=formatting(df['f'])
for prob in range(len(probs)) :
    for eq in range(len(eqs)):
        opt=hbw[(hbw['prob_id']==str(probs[prob])) & (hbw['eq']==eqs[eq])]['best_sol'].iloc[0]
        for i in df.loc[(df['problemID']==probs[prob]) & (df['eq']==eqs[eq])].index:
            point=df.loc[i,'f']
            df.loc[i,'dis']=np.linalg.norm(opt-point)
       
       
       
       
vf_means = pd.pivot_table(df, values='dis', index=['problemID','eq'],
                          columns=['interaction'], aggfunc='mean', fill_value=0)

print(vf_means.to_latex(float_format="%.2f"))

nrow=1
fig, axs = plt.subplots(1,3 ,sharex=True, sharey=False,
                        squeeze=False, gridspec_kw={'hspace': 0.1},figsize=(8,8),
                        constrained_layout=True)#

for prob in range(len(probs)) :

    # plt.figure(figsize=(6,3))
    sns.set_style("white")
    #calculation of subfig indexes
    c=prob
    r=0
    for eq in range(len(eqs)):
        axs[r][c].plot(vf_means.loc[probs[prob]].loc[eqs[eq]].index.values,vf_means.loc[probs[prob]].loc[eqs[eq]].values )
    

    # axs[r][c].set(yticklabels=[])
    axs[r][c].set_xlabel('')
    axs[r][c].margins(y=0)
    axs[r][c].annotate("DTLZ {}".format(probs[prob]),xy=(0.0,0), xycoords='axes fraction', xytext=(1.05, 0.3), textcoords='axes fraction', rotation='vertical')
    

# plt.tight_layout()
pdf_filename = "Hidden-167-distance"+suffix+".pdf"
handles, labels = axs[0][0].get_legend_handles_labels()
# l = plt.legend(handles[0:2], labels[0:2], borderaxespad=0., ncol=2), bbox_to_anchor=(0.25, 7),mode="expand"
# fig.legend(handles,labels[0:3], ncol=3, loc='upper center',borderaxespad=3)
fig.legend()
fig.text(0.5, 1.05, 'Utility value of selected solutions over interactions' , ha='center', fontsize=12)
fig.text(0.5, -.02, 'Number of interactions', ha='center',fontsize=14)
fig.text(-0.05, 0.5, 'Utility value', va='center', rotation='vertical',fontsize=14)
plt.savefig(pdf_filename, bbox_inches='tight')
plt.show()
'''