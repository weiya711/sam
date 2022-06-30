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


def create_REORDER_plots(df):
    # print(df.to_string())
    small_df = df.filter(items=['name', 'cycles', 'sparsity'])

    test_dfs = {}

    reorder_tests = ['ikj',
                     'kij',
                     'ijk',
                     'kji',
                     'jki',
                     'jik']

    for test in reorder_tests:

        # t_df = small_df.filter(like='test')
        t_df = small_df[test in small_df['name']]
        test_dfs[test] = t_df
        # print(t_df)

    # First example graph - fixed split factor, vary sparsity
    plot_reorder(test_dfs=test_dfs, test_names=reorder_tests, legend=reorder_tests)


def plot_reorder(test_dfs, test_names, legend):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test]
        plt.plot(t_df['sparsity'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.yscale('log', base=2)
    # plt.xscale('logit', base=2)
    plt.xscale('logit')
    plt.xlabel('Sparsity (% nonzeros)', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Sparsity (random)', **title_keywords)
    fig.set_size_inches(16, 12)
    plt.savefig('reorder.pdf')


def create_FUSION_plots(df):
    # print(df.to_string())
    small_df = df.filter(items=['cycles', 'vectype', 'sparsity', 'split_factor', 'test_name', 'block_size', 'run_length'])

    test_dfs = {}

    fusion_names = ['test_mat_sddmm_coiter_fused',
                    'test_mat_sddmm_locate_fused',
                    'test_mat_sddmm_unfused']

    fusion_legend = ['sddmm_coiter_fused',
                     'sddmm_locate_fused',
                     'sddmm_unfused']

    for test in fusion_names:

        # t_df = small_df.filter(like='test')
        # t_df = small_df[small_df['test_name'] == test]
        t_df = small_df[test in small_df['name']]
        test_dfs[test] = t_df
        # print(t_df)

    # First example graph - fixed split factor, vary sparsity
    plot_fusion(test_dfs=test_dfs, test_names=fusion_names, legend=fusion_legend)


def plot_fusion(test_dfs, test_names, legend):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test]
        plt.plot(t_df['sparsity'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.yscale('log', base=2)
    # plt.xscale('logit', base=2)
    plt.xscale('logit')
    plt.xlabel('Sparsity (% nonzeros)', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Sparsity (random)', **title_keywords)
    fig.set_size_inches(16, 12)
    plt.savefig('fusion.pdf')


def create_ACCEL_plots(df):
    # print(df.to_string())
    small_df = df.filter(items=['cycles', 'vectype', 'sparsity', 'split_factor', 'test_name', 'block_size', 'run_length'])

    test_dfs = {}

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

    for test in test_names:

        # t_df = small_df.filter(like='test')
        t_df = small_df[small_df['test_name'] == test]
        test_dfs[test] = t_df
        # print(t_df)

    # First example graph - fixed split factor, vary sparsity
    plot_urandom_const_sf(test_dfs=test_dfs, test_names=test_names, legend=test_names_legend)
    plot_urandom_sf_const_sp(test_dfs=test_dfs, test_names=test_names, legend=test_names_legend)
    plot_runs_sf(test_dfs=test_dfs, test_names=test_names, legend=test_names_legend)
    plot_blocks_sf(test_dfs=test_dfs, test_names=test_names, legend=test_names_legend)


def plot_urandom_const_sf(test_dfs, test_names, legend):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'random') &
                              ((test_dfs[test]['split_factor'] == 1) | (test_dfs[test]['split_factor'] == 64))]
        plt.plot(t_df['sparsity'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.yscale('log', base=2)
    # plt.xscale('logit', base=2)
    plt.xscale('logit')
    plt.xlabel('Sparsity (% nonzeros)', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Sparsity (random)', **title_keywords)
    fig.set_size_inches(16, 12)
    plt.savefig('urandom_const_sf.pdf')


def plot_urandom_sf_const_sp(test_dfs, test_names, legend):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'random') & ((test_dfs[test]['sparsity'] == 0.9))]
        print(t_df)
        plt.plot(t_df['split_factor'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.xlabel('Split Factor', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Split Factor (random)', **title_keywords)
    fig.set_size_inches(16, 12)
    plt.savefig('urandom_const_sp.pdf')


def plot_runs_sf(test_dfs, test_names, legend):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'runs') &
                              ((test_dfs[test]['split_factor'] == 1) | (test_dfs[test]['split_factor'] == 64))]
        print(t_df)
        plt.plot(t_df['run_length'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.xlabel('Run Length', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Split Factor (runs)', **title_keywords)
    fig.set_size_inches(16, 12)
    plt.savefig('runs_sf.pdf')


def plot_blocks_sf(test_dfs, test_names, legend):
    fig = plt.figure()
    for idx, test in enumerate(test_names):
        t_df = test_dfs[test][(test_dfs[test]['vectype'] == 'blocks') &
                              ((test_dfs[test]['split_factor'] == 1) | (test_dfs[test]['split_factor'] == 64))]
        print(t_df)
        plt.plot(t_df['block_size'], t_df['cycles'], marker=test_markers[idx], **plot_keywords)
    plt.legend(legend, **legend_keywords)
    plt.xlabel('Block Size', **xlabel_keywords)
    plt.ylabel('Simulator Cycles', **ylabel_keywords)
    plt.title('Simulator Cycles vs Split Factor (blocks)', **title_keywords)
    fig.set_size_inches(16, 12)
    plt.savefig('blocks_sf.pdf')


def create_ASPLOS_plots(csv_path, name=None):

    assert name is not None
    df = pd.read_csv(csv_path)

    if name == "ACCEL":
        create_ACCEL_plots(df)
    elif name == "REORDER":
        create_REORDER_plots(df)
    elif name == "FUSION":
        create_FUSION_plots(df)
    else:
        raise NotImplementedError

    # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot synthetics results csv')
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    csv_path = args.csv_path
    name = args.name

    assert csv_path is not None
    assert name is not None

    create_ASPLOS_plots(csv_path=csv_path, name=name)
