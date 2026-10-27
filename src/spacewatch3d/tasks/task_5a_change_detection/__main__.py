"""Analyze fresh mesh/Gaussian inputs or publish a prepared whole-mesh comparison."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Task 5a: mesh/Gaussian change detection')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path)
    inputs.add_argument('--manifest', type=Path, help='Fresh mesh/Gaussian/camera input manifest')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, help='Local-only GPU/model/SSH configuration')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--materials', type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args(argv)
    if args.manifest:
        from .gaussian_backend import analyze
        print(analyze(args.manifest, args.output, args.runtime, validate_only=args.validate_only))
        return 0
    from .dorm_surface import main as publish
    publish(args.source, args.output, args.materials, args.previous)
    print(args.output.resolve() / 'tour.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
