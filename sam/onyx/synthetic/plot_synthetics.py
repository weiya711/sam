import matplotlib.pyplot as plt
import matplotlib
import argparse
import pandas as pd

matplotlib.use('tkagg')
legend_keywords = {'prop': {'size': 24}}
plot_keywords = {'markersize': 18}
title_keywords = {'fontsize': 24}
xlabel_keywords = {'fontsize': 24}
ylabel_keywords = {'fontsize': 24}

test_markers = ['o', "v", "^", "s", "+", "x"]

test_names = ['test_unit_vec_elemmul_c_c_c',
              'test_unit_vec_elemmul_u_u_u',
              'test_vec_elemmul_skip_c_c_c',
              'test_vec_elemmul_split',
              'test_vec_elemmul_bv_split',
              'test_vec_elemmul_bv']

test_names_legend = ['compressed',
                     'dense',
                     'compressed_skip',
                     'compressed_split',
                     'bitvector_split',
                     'bitvector']


def create_ASPLOS_plots(csv_path):

    df = pd.read_csv(csv_path)
    # print(df.to_string())
    small_df = df.filter(items=['cycles', 'vectype', 'sparsity', 'split_factor', 'test_name'])

    test_dfs = {}

    for test in test_names:

        # t_df = small_df.filter(like='test')
        t_df = small_df[small_df['test_name'] == test]
        test_dfs[test] = t_df
        # print(t_df)

    # First example graph - fixed split factor, vary sparsity
    plot_urandom_const_sf(test_dfs=test_dfs)
    plot_urandom_sf_const_sp(test_dfs=test_dfs)
    plot_runs_sf(test_dfs=test_dfs)
    plot_blocks_sf(test_dfs=test_dfs)

    plt.show()


def plot_urandom_const_sf(test_dfs):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'random') &
                              ((test_dfs[test]['split_factor'] == 1) | (test_dfs[test]['split_factor'] == 64))]
        plt.plot(t_df['sparsity'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(test_names_legend, **legend_keywords)
    plt.yscale('log', base=2)
    # plt.xscale('logit', base=2)
    plt.xscale('logit')
    plt.xlabel('Sparsity (% nonzeros)', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Sparsity (random)', **title_keywords)


def plot_urandom_sf_const_sp(test_dfs):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'random') & ((test_dfs[test]['sparsity'] == 0.9))]
        print(t_df)
        plt.plot(t_df['split_factor'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(test_names_legend, **legend_keywords)
    plt.xlabel('Split Factor', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Split Factor (random)', **title_keywords)


def plot_runs_sf(test_dfs):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'runs')]
        print(t_df)
        plt.plot(t_df['split_factor'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(test_names_legend, **legend_keywords)
    plt.xlabel('Split Factor', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Split Factor (runs)', **title_keywords)


def plot_blocks_sf(test_dfs):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'blocks')]
        print(t_df)
        plt.plot(t_df['split_factor'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(test_names_legend, **legend_keywords)
    plt.xlabel('Split Factor', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Split Factor (blocks)', **title_keywords)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot synthetics results csv')
    parser.add_argument("--csv_path", type=str, default=None)
    args = parser.parse_args()

    csv_path = args.csv_path

    assert csv_path is not None

    create_ASPLOS_plots(csv_path=csv_path)
