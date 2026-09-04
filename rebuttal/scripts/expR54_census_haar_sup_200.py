#!/usr/bin/env python3
"""expR54 (review-response, A1): the Haar x supremum cell of the 2x2, 200 replicates.
Thin wrapper around expR52 (same engine, --null haar --stat sup); output expR54_census_haar_sup_200.csv."""
import sys, runpy
sys.argv = [sys.argv[0], "--null", "haar", "--stat", "sup"] + sys.argv[1:]
runpy.run_path(str(__import__("pathlib").Path(__file__).with_name("expR52_census_haar_p999_200.py")), run_name="__main__")
