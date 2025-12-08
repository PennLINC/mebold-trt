# which conda
# conda activate salot

# NOTE: subjects must be filled in manually, but their IDs are PII, so we cannot share them here.
# Instead, we have the anonymized subject IDs here.
declare -a subses1=("01" "02" "03" "04" "05" "06" "07" "08")
for sub in "${subses1[@]}"
do
    echo "$sub"
    heudiconv \
        -f reproin \
        -o /cbica/projects/executive_function/mebold_trt/dset \
        -d "/cbica/projects/executive_function/mebold_trt/sourcedata/{subject}_{session}/*/*/*/*.dcm" \
        -s $sub \
        -ss 1 \
        --bids
done

declare -a subses2=("02" "03" "04" "05" "06" "07" "08")
for sub in "${subses2[@]}"
do
    echo "$sub"
    heudiconv \
        -f reproin \
        -o /cbica/projects/executive_function/mebold_trt/dset \
        -d "/cbica/projects/executive_function/mebold_trt/sourcedata/{subject}_{session}/*/*/*/*.dcm" \
        -s ${sub} \
        -ss 2 \
        --bids
done

declare -a subses3=("06")
for sub in "${subses3[@]}"
do
    heudiconv \
        -f reproin \
        -o /cbica/projects/executive_function/mebold_trt/dset \
        -d "/cbica/projects/executive_function/mebold_trt/sourcedata/{subject}_{session}/*/*/*/*.dcm" \
        -s $sub \
        -ss 3 \
        --bids
done

declare -a subsesnohc=("08")
for sub	in "${subsesnohc[@]}"
do
    heudiconv \
        -f reproin \
        -o /cbica/projects/executive_function/mebold_trt/dset \
        -d "/cbica/projects/executive_function/mebold_trt/sourcedata/{subject}_{session}/*/*/*/*.dcm" \
        -s $sub \
        -ss noHC \
        --bids
done
