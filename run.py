# run.py
import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict', 'tune'], default='train')
    args = parser.parse_args()
    from rogii.pipeline import run_pipeline
    import config
    run_pipeline(config, mode=args.mode)

if __name__ == '__main__':
    main()
