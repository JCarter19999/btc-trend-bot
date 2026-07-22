import argparse, json
from datetime import datetime, timezone
import pandas as pd
from .binance import BinanceClient
from .config import load_v1_config
from .data import download_history
from .features import build_features, FEATURE_COLUMNS
from .strategy import generate_candidates
from .labeling import mature_candidate, mature_candidate_adaptive
from .model import train_logistic
from .promotion import promote
from .storage import connect

def main() -> None:
    parser = argparse.ArgumentParser(prog="btc-v1")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--start", required=True); download.add_argument("--end"); download.add_argument("--output", required=True); download.add_argument("--config", default="config/v1.yaml")
    dataset = sub.add_parser("build-dataset")
    dataset.add_argument("--candles", required=True); dataset.add_argument("--output", required=True); dataset.add_argument("--config", default="config/v1.yaml"); dataset.add_argument("--exit-engine", choices=["fixed", "adaptive"], default="adaptive")
    train = sub.add_parser("train")
    train.add_argument("--dataset", required=True); train.add_argument("--model-id", required=True); train.add_argument("--output", required=True); train.add_argument("--config", default="config/v1.yaml")
    approve = sub.add_parser("promote")
    approve.add_argument("--model", required=True); approve.add_argument("--approved-by", required=True); approve.add_argument("--reason", default=""); approve.add_argument("--config", default="config/v1.yaml")
    args = parser.parse_args()
    cfg = load_v1_config(args.config)
    if args.command == "download":
        download_history(BinanceClient(cfg["exchange"]["base_url"]), cfg["system"]["symbol"], cfg["system"]["interval"], args.start, args.end, args.output)
    elif args.command == "build-dataset":
        candles = pd.read_parquet(args.candles)
        features = build_features(candles, cfg)
        rows = []
        for candidate in generate_candidates(features, cfg):
            label = mature_candidate_adaptive(candidate, features, cfg["costs"], cfg["exit"]) if args.exit_engine == "adaptive" else mature_candidate(candidate, features, cfg["costs"])
            label.update(candidate.features)
            rows.append(label)
        pd.DataFrame(rows).to_parquet(args.output, index=False)
        print(json.dumps({"candidates": len(rows), "output": args.output}))
    elif args.command == "train":
        data = pd.read_parquet(args.dataset)
        model, metrics = train_logistic(data, args.model_id, cfg["model"]["safety_margin_bps"])
        manifest = {"model_id": args.model_id, "created_at": datetime.now(timezone.utc).isoformat(), "strategy_version": cfg["system"]["strategy_version"], "safety_margin_bps": cfg["model"]["safety_margin_bps"], "features": FEATURE_COLUMNS, "metrics": metrics, "gates": {}}
        model.save(args.output, manifest)
        print(json.dumps(metrics, indent=2))
    elif args.command == "promote":
        promote(args.model, cfg["model"]["champion_dir"], connect(cfg["storage"]["sqlite_path"]), args.approved_by, args.reason)

if __name__ == "__main__":
    main()
