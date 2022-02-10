def bmarkFile = 'run_benchmarks.jl'
def prNumber = BRANCH_NAME.tokenize("PR-")[0]
pipeline {
  agent any
  environment {
    REPO_EXISTS = fileExists "$repo"
  }
  options {
    skipDefaultCheckout true
  }
  triggers {
    GenericTrigger(
     genericVariables: [
        [
            key: 'action', 
            value: '$.action',
            expressionType: 'JSONPath', //Optional, defaults to JSONPath
            regexpFilter: '[^(created)]', //Optional, defaults to empty string
            defaultValue: '' //Optional, defaults to empty string
        ],
        [
            key: 'comment',
            value: '$.comment.body',
            expressionType: 'JSONPath', //Optional, defaults to JSONPath
            regexpFilter: '', //Optional, defaults to empty string
            defaultValue: '' //Optional, defaults to empty string
        ],
        [
            key: 'org',
            value: '$.organization.login',
            expressionType: 'JSONPath', //Optional, defaults to JSONPath
            regexpFilter: '', //Optional, defaults to empty string
            defaultValue: 'JuliaSmoothOptimizers' //Optional, defaults to empty string
        ],
        [
            key: 'pullrequest',
            value: '$.issue.number',
            expressionType: 'JSONPath', //Optional, defaults to JSONPath
            regexpFilter: '[^0-9]', //Optional, defaults to empty string
            defaultValue: '' //Optional, defaults to empty string
        ],
        [
            key: 'repo',
            value: '$.repository.name',
            expressionType: 'JSONPath', //Optional, defaults to JSONPath
            regexpFilter: '', //Optional, defaults to empty string
            defaultValue: '' //Optional, defaults to empty string
        ]
     ],

     causeString: 'Triggered on $comment',

     token: "Krylov",

     printContributedVariables: true,
     printPostContent: true,

     silentResponse: false,

     regexpFilterText: '$comment $pullrequest',
     regexpFilterExpression: '@JSOBot runbenchmarks( .*\\.jl)? ' + prNumber
    )
  }
  stages {
    stage('clone repo') {
      when {
        expression { REPO_EXISTS == 'false' && env.comment }
      }
      steps {
        sh 'git clone https://${GITHUB_AUTH}@github.com/$org/$repo.git'
      }
    }
    stage('checkout on new branch') {
      when {
        expression { env.comment }
      }
      steps {
        dir(WORKSPACE + "/$repo") {
          sh '''
            git clean -fd
            git checkout main
            git pull origin main
            git fetch origin
            LOCAL_BRANCH_NAME="temp_bmark"
            git branch -D $LOCAL_BRANCH_NAME || true
            git fetch origin pull/$pullrequest/head:$LOCAL_BRANCH_NAME
            git checkout $LOCAL_BRANCH_NAME --
          '''
        }
      }
    }
    stage('run benchmarks') {
      when {
        expression {env.comment}
      }
      steps {
        script {
          def data = env.comment.tokenize(' ')
          if (data.size() > 2) {
            bmarkFile = data.get(2);
          }
        }
        dir(WORKSPACE + "/$repo") {
          sh "mkdir -p $HOME/benchmarks/${org}/${repo}"
          sh "qsub -N ${repo}_${pullrequest} -V -cwd -o $HOME/benchmarks/${org}/${repo}/${pullrequest}_${BUILD_NUMBER}_bmark_output.log -e $HOME/benchmarks/${org}/${repo}/${pullrequest}_${BUILD_NUMBER}_bmark_error.log benchmark/push_benchmarks.sh $bmarkFile"  
        }
      }
    }
  }
  post {
    success {
      echo "SUCCESS!"  
    }
    cleanup {
      sh 'printenv'
    }
  }
}

