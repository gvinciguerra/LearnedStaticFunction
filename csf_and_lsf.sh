#!/bin/bash

(cd csf && /bin/bash initDocker.sh)
(cd csf && /bin/bash run.sh)
/bin/bash run_all_lsf.sh
