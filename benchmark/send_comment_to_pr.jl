using ArgParse
using Git
using GitHub
using JSON

DEFAULT_GIST_FILE_PATH = "gist.json"

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--org", "-o"
        help = "Name of GitHub Organization"
        arg_type = String
        default = "JuliaSmoothOptimizers"
        "--repo", "-r"
        help = "The name of the repository on GitHub"
        arg_type = String
        required = true
        "--pullrequest", "-p"
        help = "An integer that corresponds to the pull request"
        required = true
        arg_type = Int
        "--gist", "-g"
        help = "specify this argument if you want to send a gist to the pullrequest. If this option is not specified, you must specify a comment"
        arg_type = String
        required = false
        "--comment", "-c"
        help = "Comment to post on the pull request"
        arg_type = String
        required = true
    end

    return parse_args(s, as_symbols = true)
end

function create_gist_from_json_file(myauth)
    gistfile = DEFAULT_GIST_FILE_PATH
    gist = begin
        open(gistfile, "r") do f
            return JSON.parse(f)
        end
    end
    posted_gist = create_gist(params = gist, auth = myauth)
    return posted_gist
end

function create_gist_from_log_file(gist_file, pullrequest_id, myauth)
    file_content = ""
    file = open(gist_file, "r")

    file_lines = readlines(file)
    line_number = findfirst(x -> !isnothing(match(r"ERROR:", x)), file_lines)
    lines = !isnothing(line_number) ? file_lines[line_number:end] : [""]
    for line in lines
        file_content *= line * '\n'
    end
    close(file)

    file_dict = Dict("$(pullrequest_id)_bmark_error.log" => Dict("content" => file_content))
    gist = Dict{String,Any}(
        "description" => "Benchmark logs",
        "public" => true,
        "files" => file_dict,
    )

    posted_gist = GitHub.create_gist(params = gist, auth = myauth)

    return posted_gist
end

function get_repo(api::GitHub.GitHubWebAPI, org::String, repo_name::String; kwargs...)
    my_params = Dict(:visibility => "all")
    # return GitHub.repo(api, repo; params = my_params, kwargs...)
    return Repo(
        GitHub.gh_get_json(
            api,
            "/repos/$(org)/$(repo_name)";
            params = my_params,
            kwargs...,
        ),
    )
end

function get_pull_request(
    api::GitHub.GitHubWebAPI,
    org::String,
    repo::Repo,
    pullrequest_id;
    kwargs...,
)
    my_params = Dict(:sort => "popularity", :direction => "desc")
    pull_request = PullRequest(
        GitHub.gh_get_json(
            api,
            "/repos/$(org)/$(repo.name)/pulls/$(pullrequest_id)";
            params = my_params,
            kwargs...,
        ),
    )
    return pull_request
end

function post_comment_to_pr(
    org::String,
    repo_name::String,
    pullrequest_id::Int,
    comment::String;
    kwargs...,
)
    api = GitHub.DEFAULT_API
    repo = get_repo(api, org, repo_name; kwargs...)
    pull_request = get_pull_request(api, org, repo, pullrequest_id; kwargs...)
    GitHub.create_comment(api, repo, pull_request, comment; kwargs...)
end

function main()
    # Need to add GITHUB_AUTH to your .bashrc
    myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])
    # parse the arguments: 
    parsed_args = parse_commandline()
    org = parsed_args[:org]
    repo_name = parsed_args[:repo]
    pullrequest_id = parsed_args[:pullrequest]
    gist_file = parsed_args[:gist]
    comment = parsed_args[:comment]

    if !isnothing(gist_file)
        if gist_file == DEFAULT_GIST_FILE_PATH
            comment = "$(comment): $(create_gist_from_json_file(myauth).html_url)"
        else
            comment = "$(comment): $(create_gist_from_log_file(gist_file, pullrequest_id, myauth).html_url)"
        end
    end
    post_comment_to_pr(org, repo_name, pullrequest_id, comment; auth = myauth)
end

main()
