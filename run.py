# run.py
import argparse
import os

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'tune'], default='train')
    args = parser.parse_args()
    from rogii.pipeline import run_pipeline
    import config
    result = run_pipeline(config, mode=args.mode)
    if args.mode == 'predict' and result is not None and len(result):
        os.makedirs(config.DATA['submissions_dir'], exist_ok=True)
        out = os.path.join(config.DATA['submissions_dir'], 'submission.csv')
        result.to_csv(out, index=False)
        print(f'Submission saved to {out}')

if __name__ == '__main__':
    main()
