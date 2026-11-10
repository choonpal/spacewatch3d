#!/usr/bin/env python3
"""Single entrypoint: preparation, native pipeline, status, and final viewer."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def main():
    os.chdir(ROOT)
    if sys.argv[1:]==['--version']:
        print('SpaceWatch Final 2026.11.10.1 (preserved native core5.1.9)')
        return 0
    if sys.argv[1:] in (['--help'],['-h']):
        print('Additional integrated commands: prepare --prepare-request ...; run-fresh --request ... --output ...\n')
    if len(sys.argv)>1 and sys.argv[1]=='prepare':
        from spacewatch_pipeline.upstream519 import main as prepare
        return prepare(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='run-fresh':
        from spacewatch_pipeline.submission_run import main as fresh
        return fresh(sys.argv[2:])
    from spacewatch_pipeline.cli import main as native
    return native(sys.argv[1:])


if __name__=='__main__':
    raise SystemExit(main())
