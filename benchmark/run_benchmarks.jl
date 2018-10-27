using GitHub, JSON, PkgBenchmark

commit = benchmarkpkg("Krylov")  # current state of repository
# master = benchmarkpkg("Krylov", "master")
# judgement = judge(commit, master)
# export_markdown("benchmark.md", judgement)
export_markdown("benchmark.md", commit)

gist_json = JSON.parse(
			"""
            {
            "description": "A benchmark for Krylov repository",
            "public": true,
            "files": {
                "benchmark.md": {
                "content": "$(escape_string(sprint(export_markdown, commit)))"
                }
            }
            }
            """
        )

# Need to add GITHUB_AUTH to your .bashrc
myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])
posted_gist = create_gist(params = gist_json, auth = myauth)
println(posted_gist.html_url)
