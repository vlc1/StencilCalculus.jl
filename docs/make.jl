using Documenter
using StencilCalculus
using StencilCore

makedocs(;
    sitename = "StencilCalculus.jl",
    # StencilCore listed so re-exported types' docstrings resolve in @docs.
    modules = [StencilCalculus, StencilCore],
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Differentiation" => "differentiation.md",
        "API reference" => "api.md",
    ],
    checkdocs = :none,
    warnonly = [:cross_references],
)

deploydocs(;
    repo = "github.com/vlc1/StencilCalculus.jl",
    devbranch = "main",
)
