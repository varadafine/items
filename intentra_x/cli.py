"""Command-line entry point for joint-risk demonstrations and OSM updates."""

from __future__ import annotations

import argparse
import json

try:
    from .ground_risk.osm_provider import OSM_FEATURE_TAGS, OSMProvider
    from .run_joint_demo import run_demo
except ImportError:
    from ground_risk.osm_provider import OSM_FEATURE_TAGS, OSMProvider
    from run_joint_demo import run_demo


def main():
    parser = argparse.ArgumentParser(prog="intentra-x")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("joint-demo", help="run air-to-ground Hohhot demo")
    demo_parser.add_argument("--gif", action="store_true")
    demo_parser.add_argument("--output-dir", default=None)

    osm_parser = subparsers.add_parser("update-osm", help="update a cached OSM category")
    osm_parser.add_argument("category", choices=sorted(OSM_FEATURE_TAGS))
    osm_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "joint-demo":
        path, payload = run_demo(args.output_dir, make_gif=args.gif)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Saved: {path}")
    elif args.command == "update-osm":
        payload = OSMProvider().query_category(args.category, force_refresh=args.force)
        print(json.dumps({"category": args.category, "count": len(payload["elements"])}))


if __name__ == "__main__":
    main()

