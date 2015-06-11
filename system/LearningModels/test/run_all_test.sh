allTest=$( ls *.py )
for testFile in $allTest; do
    python $testFile
    rc=$?
    if [[ $rc != 0 ]]; then
        exit $rc
    fi
done
