#!/bin/bash

set -e
git clean -fd
git checkout main
git pull origin main
git fetch origin
LOCAL_BRANCH_NAME="temp_bmark"
git branch -D $LOCAL_BRANCH_NAME || true
git fetch origin pull/$pullrequest/head:$LOCAL_BRANCH_NAME
git checkout $LOCAL_BRANCH_NAME -- || true

julia --project=benchmark -E 'using Pkg; Pkg.resolve()'
julia --project=benchmark benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -c '**Starting benchmarks!**'


julia --project=benchmark benchmark/$1 $repo

if [ "$?" -eq "0" ] ; then
    julia --project=benchmark benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -c "Benchmark results" -g "gist.json"
else
    ERROR_LOGS="/home/jenkins/benchmarks/$org/$repo/${pullrequest}_${BUILD_NUMBER}_bmark_error.log"
    julia --project=benchmark benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -c "**An error occured while running $1**" -g $ERROR_LOGS
fi

git clean -fd
git reset --hard
git checkout main
