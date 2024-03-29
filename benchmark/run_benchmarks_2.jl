gist_json = JSON.parse("""
    {
        "description": "A benchmark for Krylov repository",
        "public": true,
        "files": {
            "judgement.md": {
                "content": "$(escape_string(sprint(export_markdown, judgement)))"
            },
            "main.md": {
                "content": "$(escape_string(sprint(export_markdown, main)))"
            },
            "commit.md": {
                "content": "$(escape_string(sprint(export_markdown, commit)))"
            }
        }
    }""")

# Need to add GITHUB_AUTH to your .bashrc
myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])
posted_gist = create_gist(params = gist_json, auth = myauth)
println(posted_gist.html_url)
