#!/usr/bin/env python3
"""Connect the validated Dorm3 5.1.8 run from your PC."""
import sys
from connect_viewer import main

if __name__=='__main__':
    sys.argv[1:1]=['--run','/workspace/spacewatch_runs/dorm3_pipeline_v5_1_8_20261026']
    main()
