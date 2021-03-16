using Git, GitHub, JSON, Pkg

"""
    test_breakage_deploy()

Read files from breakage-info and publish to the PR
"""
function test_breakage_deploy()
  if get(ENV, "CI", "false") == "false"
    error("Only run on CI")
  end
  if get(ENV, "GITHUB_AUTH", nothing) === nothing
    error("GITHUB_AUTH not found")
  end
  key = ENV["GITHUB_AUTH"]
  repo = "JuliaSmoothOptimizers/Krylov.jl"
  user = split(repo, "/")[1]
  upstream = "https://$user:$key@github.com/$repo"
  Git.run(`config user.email "abel.s.siqueira@gmail.com"`)
  Git.run(`config user.name "Abel Soares Siqueira"`)
  Git.run(`remote add upstream $upstream`)
  Git.run(`fetch upstream`)
  Git.run(`checkout -f breakage-info`)

  badge_pass(x) = "![](https://img.shields.io/badge/$x-Pass-green)"
  badge_fail(x) = "![](https://img.shields.io/badge/$x-Fail-red)"
  badge(tf, x) = tf ? badge_pass(x) : badge_fail(x)

  packages = ["CaNNOLeS", "DCI", "JSOSolvers", "Percival"]

  output = ":robot: Testing breakage of this pull request\n\n"
  output *= "| Package Name | master | stable |\n"
  output *= "|--|--|--|\n"
  for package in packages
    output *= "| $package | "

    for version in ["master", "stable"]
      info = JSON.parse(open("$package-$version"))
      bdg = badge(info["pass"], info["tag"])
      joburl = info["joburl"]
      output *= "[$bdg]($joburl) | "
    end
    output *= "\n"
  end

  @debug(output)

  myauth = GitHub.authenticate(key)
  myrepo = GitHub.repo(repo, auth=myauth) # "JuliaSmoothOptimizers/Krylov.jl"
  prs = pull_requests(myrepo, auth=myauth)
  pr = nothing
  prnumber = ENV["GITHUB_REF"]
  @debug("PR NUMBER: $prnumber")
  prnumber = split(prnumber, "/")[3]

  for p in prs[1]
    if p.number == Meta.parse(prnumber)
      pr = p
    end
  end
  @assert pr != nothing

  GitHub.create_comment(GitHub.DEFAULT_API, myrepo, pr, :pr, auth=myauth, params=Dict(:body => output))
end

test_breakage_deploy()
