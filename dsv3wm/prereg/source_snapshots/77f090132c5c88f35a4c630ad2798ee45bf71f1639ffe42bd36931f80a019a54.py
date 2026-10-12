"""Plot completed fixed development results without inferred or missing values."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.svg').exists():
        raise FileExistsError('figure exists')
    names = ['B0', 'A0', 'A1', 'A2', 'A5']
    files, records = [], {}
    for name in names:
        dynamics = args.results / ('B0_qualification.json' if name == 'B0' else name + '_dynamics.json')
        planning, behavior = args.results / (name + '_planning.json'), args.results / (name + '_retention.json')
        files += [dynamics, planning, behavior]
        values = [json.loads(path.read_text(encoding='utf-8')) for path in (dynamics, planning, behavior)]
        if any(value['status'] != 'development-evaluated' for value in values):
            raise ValueError('all native measurements must be complete')
        records[name] = values
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    colors = ['#77818e', '#2463a6', '#cb7728', '#7852ad', '#348875']
    x = list(range(len(names)))
    def count_bars(axis, values, total, title):
        axis.bar(x, [100 * value / total for value in values], color=colors, width=.62)
        for position, value in enumerate(values):
            axis.text(position, 100 * value / total + 2, f'{value}/{total}', ha='center', fontsize=10)
        axis.set(xticks=x, xticklabels=names, ylim=(0, 112), yticks=[0, 25, 50, 75, 100], ylabel='Percent', title=title)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    count_bars(axes[0, 0], [records[name][0]['summary']['all']['4']['rollout_correct'] for name in names], 24,
               'Four-step exact state: own-state rollout')
    adapted = names[1:]
    for offset, field, color, label in ((-.18, 'direct_correct', '#d36b58', 'Exact state'),
                                        (.18, 'direct_malformed', '#4c9a9a', 'Complete reply schema')):
        values = [records[name][0]['summary']['all']['4'][field] for name in adapted]
        if field == 'direct_malformed':
            values = [24 - value for value in values]
        positions = [index + offset for index in range(4)]
        axes[0, 1].bar(positions, [100 * value / 24 for value in values], width=.34, color=color, label=label)
        for position, value in zip(positions, values):
            axes[0, 1].text(position, 100 * value / 24 + 2, f'{value}/24', ha='center', fontsize=9)
    axes[0, 1].set(xticks=range(4), xticklabels=adapted, ylim=(0, 112), yticks=[0, 25, 50, 75, 100],
                    ylabel='Percent', title='Direct four-step prediction: state versus format')
    axes[0, 1].legend(loc='lower right', fontsize=9)
    axes[0, 1].grid(axis='y', alpha=.15)
    axes[0, 1].set_axisbelow(True)
    count_bars(axes[1, 0], [records[name][1]['summary']['all']['successes'] for name in names], 12,
               'Goal-only planning: 30 queries per task')
    count_bars(axes[1, 1], [sum(row['correct'] for row in records[name][2]['rows'] if row['domain'] == 'register')
                            for name in names], 3, 'Fixed RegisterWorld retained behavior')
    fig.suptitle('Registered-interface recovery: no predictive gain at the fixed gate', fontsize=14, y=.97)
    fig.text(.5, .035, 'One training seed; development only. Planning is nonterminating goal achievement. '
             'Retention uses three fixed register cases; all three ledger cases failed for every arm.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.07, right=.98, bottom=.12, top=.89, hspace=.38, wspace=.28)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    fig.savefig(args.out.with_suffix('.svg'))
    plt.close(fig)
    metadata = {'scope': 'fixed completed development counts; no uncertainty/confirmation or broad-retention claim',
                'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'matplotlib': matplotlib.__version__}
    with args.out.with_suffix('.sources.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    print(str(args.out), flush=True)


if __name__ == '__main__':
    main()
