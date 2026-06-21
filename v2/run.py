#!/usr/bin/env python3
"""CLI entry point for v2 pipeline."""
import argparse
import sys
sys.path.insert(0, 'v2')
from rogii_v2.pipeline import run_pipeline
import config as cfg

def main():
    parser = argparse.ArgumentParser(description='ROGII v2 Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    parser.add_argument('--train-dir', type=str, default=None)
    parser.add_argument('--test-dir', type=str, default=None)
    parser.add_argument('--models-dir', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    run_pipeline(cfg, mode=args.mode,
                 train_dir=args.train_dir, test_dir=args.test_dir,
                 models_dir=args.models_dir, output_path=args.output)

if __name__ == '__main__':
    main()
