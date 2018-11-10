using GitHub, JSON, PkgBenchmark

commit = benchmarkpkg("Krylov")  # current state of repository
master = benchmarkpkg("Krylov", "master")
judgement = judge(commit, master)
export_markdown("judgement.md", judgement)
export_markdown("benchmark-commit.md", commit)
export_markdown("benchmark-master.md", master)

gist_json = JSON.parse(
			"""
            {
	            "description": "A benchmark for Krylov repository",
	            "public": true,
	            "files": {
	                "judgement.md": {
		                "content": "$(escape_string(sprint(export_markdown, judgement)))"
	                },
					"benchmark-commit.md": {
		                "content": "$(escape_string(sprint(export_markdown, commit)))"
	                },
					"benchmark-master.md": {
		                "content": "$(escape_string(sprint(export_markdown, master)))"
	                }
	            }
            }
            """
        )

# Need to add GITHUB_AUTH to your .bashrc
myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])
posted_gist = create_gist(params = gist_json, auth = myauth)
println(posted_gist.html_url)
