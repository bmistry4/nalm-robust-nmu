import itertools
import os
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE

figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')

np.random.seed(0)


def shell_script_cmd():
    # generating marc bash cmd (chained)
    cmd = 'bash marc-baseline-nauW-grid.sh'
    l = []
    for s in range(67, 6561, 25):
        l.append(f'{cmd} {s} {s + 24} 4 0.25 0 ; ')
    ''.join(l)
    return l


def generate_intial_W(id):
    # generate permutations
    W_permutations = ([p for p in itertools.product([-0.5, 0, 0.5], repeat=8)])
    return list(W_permutations[id])  # don't reshape


def plot_2d(df):
    cluster_colors = {0: '#d95f02', 1: '#1b9e77'}
    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(df['W_init'].tolist())  # n_jobs=14
    X1, X2 = X_embedded[:, 0], X_embedded[:, 1]
    df['x1'] = X1
    df['x2'] = X2
    # df.to_pickle('/data/bm4g15/nalu-stable-exp/figures/grid_nau_W_tSNE_df_FINAL.pkl'); print('saved df with dim reduction feats')

    # plt.title('t-SNE of NAU weight initalisations with colouring to indicate success')
    plt.xlabel('x1')
    plt.ylabel('x2')

    groups = df.groupby('pass')
    for g_val, group in groups:
        plt.plot(group.x1, group.x2, marker='o', linestyle='', ms=2,
                 label=g_val, color=cluster_colors[g_val], mec='none')

    # for i in range(len(df)):
    #    plt.text(df.iloc[i]['x1'], df.iloc[i]['x2'], df.iloc[i]['seed'], size=4)

    # TODO - https://intoli.com/blog/resizing-matplotlib-legend-markers/
    plt.legend(markerscale=6)


def create_df(directory, save_path=False):
    # create df containing: seed, nau_W, result

    seeds = []
    W_inits = []
    passes = []
    for filename in os.listdir(directory):
        if filename.endswith(".out"):
            fp = os.path.join(directory, filename)
        # fp = r'C:\Users\mistr\Documents\SOTON\PhD\Code\nalu-stable-exp\tensorboard\test\test.log'

        with open(fp) as f:
            data = f.readlines()
            stats_line = data[-1]  # get line with seed and result

        seed_result_dict = literal_eval(stats_line)
        seeds.append(seed_result_dict['seed'])
        W_inits.append(generate_intial_W(seed_result_dict['seed']))
        passes.append(seed_result_dict['pass'])

    df = pd.DataFrame({'seed': seeds, 'W_init': W_inits, 'pass': passes})

    if save_path is not False:
        df.to_pickle(save_path)
        print(f'pkl saved to: {save_path}')

    return df


def plot_W_init_success_headmap(df, save_path):
    # heatmap of the 3 different init values and their success rate wrt each input feature
    plot = sns.heatmap(df, annot=True)
    plot.get_figure().savefig(save_path, dpi=400)
    print('heatmap saved')


def generate_W_init_success_df(df, save_path):
    """
       Calc stats for each NAU W feature. Calc prob of success given the value is set to one of the init values (i.e. 
       {-0.5,0,0.5}. Do this for all features 
       """
    X_col_names = ['x11', 'x12', 'x13', 'x14', 'x21', 'x22', 'x23', 'x24']
    # expand out init vec to separate cols
    df[X_col_names] = pd.DataFrame(df.W_init.tolist(), index=df.index)
    X_org_stats = []
    for col in X_col_names:
        # % of success's f.e. input feature conditioned on it's starting value -> P(success| init for x_i = {-0.5,0,0.5})
        temp = df.groupby([col]).sum()['pass'] / df.groupby([col]).count()['pass'] * 100
        temp = temp.rename(col)
        X_org_stats.append(temp)

    W_init_success_df = pd.concat(X_org_stats, axis=1, sort=False)
    print("Success given W feature is initalised to either {-0.5, 0, 0.5}")
    print(W_init_success_df)

    W_init_success_df.to_pickle(save_path)
    print('saved W_init_success_df as pkl')
    return W_init_success_df


def df_eda(results_path):
    # basic data analysis on success rates on the results
    # results_path = '/data/bm4g15/nalu-stable-exp/figures/grid_nau_W_tSNE_df_FINAL.pkl'  # TODO
    df = pd.read_pickle(results_path)

    print(f'Total runs: {df.shape[0]}')
    pass_stats = df['pass'].value_counts(normalize=True)
    print(f'Passed (%): {pass_stats[1] * 100:.3f}')
    print(f'Failed (%): {pass_stats[0] * 100:.3f}')

    save_path = '/data/bm4g15/nalu-stable-exp/figures/W_inits_success_df.pkl'
    W_init_success_df = pd.read_pickle(save_path)
    # W_init_success_df = generate_W_init_success_df(df, save_path)
    plot_W_init_success_headmap(W_init_success_df,
                                save_path='/data/bm4g15/nalu-stable-exp/figures/W_inits_success_heatmap.png')


def df_correlation_stats_and_plots():
    # Load df with the low dim tSNE embeddings
    df_path = r"C:\Users\mistr\Desktop\temp-backups\grid_nau_W_tSNE_df_FINAL.pkl"
    df = pd.read_pickle(df_path)
    X_col_names = ['x11', 'x12', 'x13', 'x14', 'x21', 'x22', 'x23', 'x24']
    df[X_col_names] = pd.DataFrame(df.W_init.tolist(), index=df.index)  # expand out init vec to separate cols

    # All failures -> where all elems init to same value
    # df.query('x11==0 & x12==0 & x13==0 & x14==0 & x21==0 & x22==0 & x23==0 & x24==0')
    # df.query('x11==0.5 & x12==0.5 & x13==0.5 & x14==0.5 & x21==0.5 & x22==0.5 & x23==0.5 & x24==0.5')
    # df.query('x11==-0.5 & x12==-0.5 & x13==-0.5 & x14==-0.5 & x21==-0.5 & x22==-0.5 & x23==-0.5 & x24==-0.5')

    def calc_condition_prob(df, query):
        q_df = df.query(query)
        return q_df[q_df['pass'] == 1].shape[0] / q_df.shape[0]

    q1 = 'x13==0 & x14==0 & x23==0 & x24==0'
    q1_prob = calc_condition_prob(df, q1)
    q2 = 'x13==0.5 & x14==0.5 & x23==0.5 & x24==0.5'
    q2_prob = calc_condition_prob(df, q2)
    q3 = 'x13==-0.5 & x14==-0.5 & x23==-0.5 & x24==-0.5'
    q3_prob = calc_condition_prob(df, q3)
    q4 = 'x11==-0.5 & x12==-0.5'
    q4_prob = calc_condition_prob(df, q4)

    print(q1, q1_prob)
    print(q2, q2_prob)
    print(q3, q3_prob)
    print(q4, q4_prob)

    def plot_2_feat_success_heatmap(df, feat1, feat2):
        probs = []
        for i in [-0.5, 0, 0.5]:
            l = []
            for j in [-0.5, 0, 0.5]:
                # feat1 = rows, and feat2= cols in dataframe
                q = f'{feat1}=={i} & {feat2}=={j}'
                p = calc_condition_prob(df, q)
                l.append(p)
                # print(q, p)
            probs.append(l)

        cond_joint_probs = np.array(probs)
        cond_joint_probs = pd.DataFrame(cond_joint_probs, index=['-0.5', '0', '0.5'], columns=['-0.5', '0', '0.5'])
        sns.heatmap(cond_joint_probs, annot=True)
        plt.xlabel(feat2)
        plt.ylabel(feat1)
        plt.show()

    plot_2_feat_success_heatmap(df, feat1='x11', feat2='x12')
    plot_2_feat_success_heatmap(df, feat1='x11', feat2='x22')
    plot_2_feat_success_heatmap(df, feat1='x11', feat2='x21')

    # show correlation for feature values for the SUCCESSFUL runs only
    df_p = df[df['pass'] == 1]
    sns.heatmap(df_p[X_col_names].corr(), annot=True)
    plt.show()

    # show correlation for feature values for the FAILED runs only
    df_p = df[df['pass'] == 0]
    sns.heatmap(df_p[X_col_names].corr(), annot=True)
    plt.show()


if __name__ == '__main__':
    logs_folder = '/data/bm4g15/nalu-stable-exp/logs/FTS_NAU_NMU_ranges/nauW-grid/nau-Nnmu-seededDS0-I4-S0.25-O0/'  # TODO
    results_path = '/data/bm4g15/nalu-stable-exp/figures/grid_nau_W_results_FINAL.pkl'  # TODO

    if os.path.exists(results_path):
        df = pd.read_pickle(results_path)
        print(f'pkl loaded from {results_path}')
    else:
        df = create_df(logs_folder, save_path=results_path)
    plot_2d(df)
    # plt.show()
    plt.savefig('/data/bm4g15/nalu-stable-exp/figures/grid_nau_W_tSNE_RS0_test.png')  # TODO
    print('plot saved')

# df[df['x1'].between(-22, -17) & df['x2'].between(17,24)].sort_values(['x1','x2'])
