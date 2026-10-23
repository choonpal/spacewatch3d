"""Run the prepared Dorm whole-mesh comparison from the installed package."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Task 5a: geometry-first Dorm surface regions')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--materials', type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args(argv)
    from .dorm_surface import main as publish
    publish(args.source, args.output, args.materials, args.previous)
    print(args.output.resolve() / 'tour.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
