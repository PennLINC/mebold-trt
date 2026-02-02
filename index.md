<br>
<br>

### Abstract


### Project Lead
Taylor Salo

### Faculty Lead
Theodore D. Satterthwaite

### Analytic Replicator
Tien Tong

### Collaborators
List to be collected

### Project Start Date
August 2023

### Current Project Status
Manuscript in preparation

### Datasets


### Github Repository
<https://github.com/PennLINC/mebold-trt>
### Slack Channel:
#mebold-abcd-compare

### Cubic Project Directory Overview
`/cbica/projects/executive_function/mebold_trt/`

<br>
<br>

# CODE DOCUMENTATION

All project analyses are described below along with the corresponding code on Github. The following outline describes the order of the analytic workflow:

*0.* Download data from FlyWheel and curate for OpenNeuro
*1.* Download from OpenNeuro
*2.* Run processing scripts



In each of the scripts, there is a variable `REPRO_DIR` or `repro_dir` that needs to be set to the directory where you're running the replication from.

<br>

### 0. Download data from FlyWheel

The steps to create the initial BIDS data can be found in [`processing_code/curation`](https://github.com/PennLINC/mebold-trt/tree/main/curation).
Briefly, the [steps](https://github.com/PennLINC/mebold-trt/tree/main/curation) were to download the dicom tarballs from Flywheel and run heudiconv.
Because no-RF images are included in the dicoms, these were split into separate files.
The anatomical scans were refaced using AFNI.
Finally, the events files were converted to BIDS format.

The validated BIDS data were uploaded to OpenNeuro.
This is where the replication process can begin -
with the datalad cloned BIDS data from OpenNeuro.

### 1. Download from OpenNeuro

As the `executivefunction` user on cubic we cloned the data from OpenNeudo using

```bash
micromamba activate babs
pip install babs==0.5.4

cd /cbica/projects/executive_function/mebold_trt
datalad install https://github.com/OpenNeuroDatasets/ds005250.git

cd /cbica/projects/executive_function/mebold_trt/ds005250
datalad get -J 10 .
```

Configure the remote S3 repository to have the highest cost, ensuring that files from the local repo are always preferred and used first. Edit the `.git/config`:

```bash
[remote "s3-PUBLIC"]
	annex-s3 = true
	annex-uuid = 683c9a58-3e80-42a2-b641-01d5becfcdda
	annex-cost = 500.0 # add this line
```

### 2. Run processing scripts

#### 2.1. MRIQC v24.0.2

Pull the Apptainer image and create a DataLad dataset containing the image

```bash
apptainer_path=/cbica/projects/executive_function/mebold_trt/software/apptainer

apptainer build \
    ${apptainer_path}/mriqc-24-0-2.sif \
    docker://nipreps/mriqc:24.0.2


apptainerDS_path=/cbica/projects/executive_function/mebold_trt/software/apptainer-ds

datalad create -D "Create mriqc-24-0-2 DataLad dataset" ${apptainerDS_path}/mriqc-24-0-2-ds
cd ${apptainerDS_path}/mriqc-24-0-2-ds
datalad containers-add \
    --url ${apptainer_path}/mriqc-24-0-2.sif \
    mriqc-24-0-2
```

Run MRIQC with BABS using the following [scripts](https://github.com/PennLINC/mebold-trt/tree/main/processing/MRIQC):

```bash
# create the BABS project
bash babs_mriqc_init.sh

# Run BABS
cd /cbica/projects/executive_function/mebold_trt/derivatives/mriqc_babs_project
# Run `babs submit` and `babs status`
# Run `babs merge` when all jobs finish successfully

# Create an ephemeral clone + unzip outputs
bash babs_mriqc_finish.sh
```

#### 2.2. NORDIC (v0.0.1) + fMRIPrep (v25.2.3)

Pull the Apptainer image and create a DataLad dataset containing the image

```bash
apptainer_path=/cbica/projects/executive_function/mebold_trt/software/apptainer

apptainer build \
    ${apptainer_path}/fmriprep-25-2-3.sif \
    docker://nipreps/fmriprep:25.2.3


cd /cbica/projects/executive_function/mebold_trt/software
git clone https://github.com/nipreps/NORDIC_Raw.git
cd NORDIC_Raw
git checkout executable
apptainer build ${apptainer_path}/nordic-0-0-1.sif Apptainer.def

###############################################################################

apptainerDS_path=/cbica/projects/executive_function/mebold_trt/software/apptainer-ds

datalad create -D "Create nordic-0-0-1 and fmriprep-25-2-3 DataLad dataset" ${apptainerDS_path}/nordic-fmriprep-ds
cd ${apptainerDS_path}/nordic-fmriprep-ds

datalad containers-add \
    --url ${apptainer_path}/fmriprep-25-2-3.sif \
    fmriprep-25-2-3

datalad containers-add \
    --url ${apptainer_path}/nordic-0-0-1.sif \
    nordic-0-0-1
```

Run NORDIC+fMRIPrep with BABS using the following [scripts](https://github.com/PennLINC/mebold-trt/tree/main/processing/NORDIC-fMRIPrep):

Create the BABS project by running [`bash babs_nordic_fmriprep_init.sh`](https://github.com/PennLINC/mebold-trt/blob/main/processing/NORDIC-fMRIPrep/babs_nordic_fmriprep_init.sh). Once the project is create, add [`fmriprep_filter.json`](https://github.com/PennLINC/mebold-trt/blob/main/processing/NORDIC-fMRIPrep/fmriprep_filter.json) and [`nordic_filter.json`](https://github.com/PennLINC/mebold-trt/blob/main/processing/NORDIC-fMRIPrep/fmriprep_filter.json) to `nordic_fmriprep_babs_project/analysis/code`, and push the filter files.

```bash
cp \
  /cbica/projects/executive_function/mebold_trt/derivatives/code/fmriprep_filter.json \
  /cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_babs_project/analysis/code
cp \
  /cbica/projects/executive_function/mebold_trt/derivatives/code/nordic_filter.json \
  /cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_babs_project/analysis/code

cd /cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_babs_project
babs sync-code -m "add nordic and fmriprep filter files"

###############################################################################

# Run BABS
cd /cbica/projects/executive_function/mebold_trt/derivatives/nordic_fmriprep_babs_project
# Run `babs submit` and `babs status`
# Run `babs merge` when all jobs finish successfully

# Create an ephemeral clone + unzip outputs
bash babs_nordic_fmriprep_finish.sh
```

#### 2.3. tedana

Run tedana using the following [scripts](https://github.com/PennLINC/mebold-trt/tree/main/processing):

```bash
# create subject-pair tsv
python generate_sub_ses_pairs.py

# Run tedana
cd /cbica/projects/executive_function/mebold_trt/github/parker/processing
sbatch run_tedana.sbatch
```

#### 2.4. XCP-D v0.13.0

Pull the Apptainer image and create a DataLad dataset containing the image

```bash
apptainer_path=/cbica/projects/executive_function/mebold_trt/software/apptainer

apptainer build \
    ${apptainer_path}/xcpd-0-13-0.sif \
    docker://pennlinc/xcp_d:0.13.0


###############################################################################

apptainerDS_path=/cbica/projects/executive_function/mebold_trt/software/apptainer-ds

datalad create -D "Create xcpd-0-13-0 DataLad dataset" ${apptainerDS_path}/xcpd-0-13-0-ds
cd ${apptainerDS_path}/xcpd-0-13-0-ds
datalad containers-add \
    --url ${apptainer_path}/xcpd-0-13-0.sif \
    xcpd-0-13-0
```

Run XCP-D with BABS using the following [scripts](https://github.com/PennLINC/mebold-trt/tree/main/processing/XCP-D):

For [single-echo](https://github.com/PennLINC/mebold-trt/tree/main/processing/XCP-D/single-echo):

```bash
# create the BABS project
bash babs_xcpd_SE_init.sh

# add filter file
cp \
	/cbica/projects/executive_function/mebold_trt/derivatives/code/xcpd_SE_filter.json \
	/cbica/projects/executive_function/mebold_trt/derivatives/xcpd_SE_babs_project/analysis/code
cd /cbica/projects/executive_function/mebold_trt/derivatives/xcpd_SE_babs_project
babs sync-code -m "Add xcpd SE filter file"

# Run BABS
cd /cbica/projects/executive_function/mebold_trt/derivatives/xcpd_SE_babs_project
# Run `babs submit` and `babs status`
# Run `babs merge` when all jobs finish successfully

# Create an ephemeral clone + unzip outputs
bash babs_xcpd_SE_finish.sh
```

For [multi-echo](https://github.com/PennLINC/mebold-trt/tree/main/processing/XCP-D/multi-echo):

First, need to `nano /cbica/projects/executive_function/mebold_trt/derivatives/tedana/dataset_description.json`
```json
{
    "Name": "tedana",
    "BIDSVersion": "1.5.0",
    "DatasetType": "derivative",
    "GeneratedBy": [{"Name": "tedana"}]
}
```

Then, save `tedana` output as a datalad dataset

```bash
cd /cbica/projects/executive_function/mebold_trt/derivatives/
datalad create -c text2git tedana_dl
cp -r tedana/* tedana_dl
datalad save \
	-d /cbica/projects/executive_function/mebold_trt/derivatives/tedana_dl \
	-m "checked dataset into datalad"
```

Lastly, use BABS to run XCP-D

```bash
# create the BABS project
bash babs_xcpd_ME_init.sh

# add filter file and tedana_xcp_d_config
cp \
	/cbica/projects/executive_function/mebold_trt/derivatives/code/xcpd_ME_filter.json \
	/cbica/projects/executive_function/mebold_trt/derivatives/xcpd_ME_babs_project/analysis/code
cp \
	/cbica/projects/executive_function/mebold_trt/derivatives/code/tedana_xcp_d_config.yml \
	/cbica/projects/executive_function/mebold_trt/derivatives/xcpd_ME_babs_project/analysis/code
cd /cbica/projects/executive_function/mebold_trt/derivatives/xcpd_ME_babs_project
babs sync-code -m "add xcpd ME filter file and tedana_xcp_d_config"

# Run BABS
cd /cbica/projects/executive_function/mebold_trt/derivatives/xcpd_ME_babs_project
# Run `babs submit` and `babs status`
# Run `babs merge` when all jobs finish successfully

# Create an ephemeral clone + unzip outputs
bash babs_xcpd_ME_finish.sh
```

#### 2.5. Fractal n-back GLMs with Nilearn

```bash
micromamba activate meboldtrt

cd /cbica/projects/executive_function/mebold_trt/github/parker/analysis

# Run first-level (run-wise) GLMs with tedana confounds
python run_nback_first_level_rtdur.py
# Run second-level (group-level) GLMs from outputs from first-level GLMs
python run_nback_second_level_rtdur.py
# Plot second-level results
python plot_nback_second_level.py
```
