#!/bin/bash
# NOTE: subjects must be filled in manually, but their IDs are PII, so we cannot share them here.
subjects=""
token=$(</path/to/flywheel_api_token.txt)
fw login $token
cd /cbica/projects/executive_function/mebold_trt/sourcedata

for subject in $subjects; do
    fw download --yes --zip fw://bbl/MEBOLD_ABCD_COMPARE/${subject}
done
