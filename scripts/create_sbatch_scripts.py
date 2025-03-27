import itertools

if __name__ == '__main__':
    servers = ['protagoras', 'protagoras', 'socrates']
    arguments = {
        'e': 'wine_test_imputers_scores',
        'ds': 'wine',
        'im': ['drop', 'mean', 'knn', 'it_br', 'it_knn', 'it_lr'],
        's': ['none', 'basic', 'double_threshold', 'range_condition', 'nonlinear'],
        'mr': [0.1*i for i in range(1, 10)],
        'rs': [i for i in range(10)]
    }
    
    script_path = "scripts/run_imputations_script.sh"
    batches_path = "scripts/run_imputations_batches.sh"
    prefix = ["sbatch", "-p", "bml", "-A", "bml"]

    flags = {k: v for k, v in arguments.items() if isinstance(v, bool)}
    params = {k: v for k, v in arguments.items() if isinstance(v, list)}
    constants = {k: v for k, v in arguments.items() if k not in flags and k not in params and v is not None}

    # Generate all combinations of parameters
    combinations = list(itertools.product(*params.values()))
    print(f'Creating {len(combinations)} tasks')

    with open(batches_path, 'w') as f:
        for i, combo in enumerate(combinations):
            server = ["-w", f"{servers[i % len(servers)]}"]
            args = constants.copy()
            args.update(dict(zip(params.keys(), combo)))
            args = {k: v for k, v in args.items() if v is not None}
            args = [[f'--{k}', f'{v}'] for k, v in args.items()] + [[f'--{k}'] for k, v in flags.items() if v]
            args = list(itertools.chain(*args))
            command = prefix + server + [script_path] + args
            f.write(' '.join(command) + '\n')
