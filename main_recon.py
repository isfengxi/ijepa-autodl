# main_recon.py
import argparse, yaml
from src.train_recon import main

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fname", type=str, required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.fname, "r"))
    main(cfg)
