######################################################################
# Readme
######################################################################
# Execute this job:
#   - connect to `nic` via ssh: `ssh username@nic` (enter passwd)
#   - start job: `condor_submit /path/to/this/file.tbi`
# 
# Monitor jobs:
#   - see machines: `condor_status`
#   - see queue: `condor_q`
#   - keep monitoring queue: `watch condor_q` (quit with ctrl + c)
# 
# Find out more at:
# http://www.iac.es/sieinvens/siepedia/pmwiki.php?n=HOWTOs.CondorHowTo
######################################################################


######################################################################
# Necessary parameters
######################################################################

# Shell script that you want to execute
cmd = /work/scratch/zzheng/github/tcl/condor/scripts/run_eval.sh 

# start directory
initialdir = /work/scratch/zzheng/

# define output, error and log file
identifier = training
output = /work/scratch/zzheng/logs/loc_$(cluster).$(Process)_$(identifier)_out.log
error = /work/scratch/zzheng/logs/loc_$(cluster).$(Process)_$(identifier)_err.log
log = /work/scratch/zzheng/logs/loc_$(cluster).$(Process)_$(identifier)_log.log

# working environments
getenv        = True
environment   = "working_dir=/work/scratch/zzheng/github/tcl/"


######################################################################
# Optional parameters
######################################################################

## A nice job will note change your priority. You can use this statement when you have enough time to wait for your results
nice_user = False

# Choose if jobs should run on cluster or workstation nodes. If unset jobs will run on each available node. Options are "cluster" or "workstations"
# required GPU RAM (MB)
# requirements = (POOL=="cluster") && (GPURAM>23000) && (GPURAM<35000) && (TARGET.Machine != "gauss.lfb.rwth-aachen.de")
requirements = (GPURAM>13000) && (GPURAM<24000) 
# requirements = (GPURAM>23000) && (GPURAM<35000)
# request a certain machine
# requirements = (TARGET.Machine=="curie.lfb.rwth-aachen.de") & (TARGET.Machine=="abacus.lfb.rwth-aachen.de") & (TARGET.Machine=="fermi.lfb.rwth-aachen.de") & (TARGET.Machine=="newton.lfb.rwth-aachen.de")
# requirements = TARGET.Machine=="fermi.lfb.rwth-aachen.de" 
# requirements = TARGET.Machine=="newton.lfb.rwth-aachen.de" 
 
# required CPU RAM
request_memory = 40 GB

# required number of CPU cores
request_cpus = 1

# required number of GPUs
request_gpus = 1

# criterion after which to choose the machine
# e.g. `rank = memory` takes machine with larges RAM
# rank = load

# number of seconds to wait before executing job 
# deferral_time = (CurrentTime + 1)

# Job Batch Name
JobBatchName = "tcl_eval"

######################################################################
# Further preferences
######################################################################

# sync logfile to logfiles instead of copying them after finishing
stream_error = true
stream_output = true
should_transfer_files = YES

# run with user's account
run_as_owner = True
load_profile = True

# send email notifications (Always|Error|Complete|Never)
notify_user   = zzheng@lfb.rwth-aachen.de
notification  = Always

# number of executions of this job
queue 1 arguments from (
#/work/scratch/zzheng/github/tcl/configs/tcl_threshold.yml --tag eval-tcl-th_0.0 --threshold 0.0
#/work/scratch/zzheng/github/tcl/configs/tcl_threshold.yml --tag eval-tcl-th_0.1 --threshold 0.1
#/work/scratch/zzheng/github/tcl/configs/tcl_threshold.yml --tag eval-tcl-th_0.2 --threshold 0.2
#/work/scratch/zzheng/github/tcl/configs/tcl_threshold.yml --tag eval-tcl-th_-0.1 --threshold -0.1
#/work/scratch/zzheng/github/tcl/configs/tcl_threshold.yml --tag eval-tcl-th_-0.2 --threshold -0.2
#/work/scratch/zzheng/github/tcl/configs/tcl_quantile.yml --tag eval-tcl-qt_0.4 --quantile 0.4
#/work/scratch/zzheng/github/tcl/configs/tcl_quantile.yml --tag eval-tcl-qt_0.5 --quantile 0.5
#/work/scratch/zzheng/github/tcl/configs/tcl_quantile.yml --tag eval-tcl-qt_0.6 --quantile 0.6
#/work/scratch/zzheng/github/tcl/configs/tcl_quantile.yml --tag eval-tcl-qt_0.7 --quantile 0.7
#/work/scratch/zzheng/github/tcl/configs/tcl_quantile.yml --tag eval-tcl-qt_0.8 --quantile 0.8
#/work/scratch/zzheng/github/tcl/configs/tcl_quantile.yml --tag eval-tcl-qt_0.9 --quantile 0.9
#/work/scratch/zzheng/github/tcl/configs/tcl_mask_emb.yml --tag eval-tcl-maskemb
/work/scratch/zzheng/github/tcl/configs/tcl_mask_img.yml --tag eval-tcl-maskimg
)
