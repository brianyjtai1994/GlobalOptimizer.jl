module GlobalOptimizer

mutable struct Agent
    x::Vector{Float64} # parameters passed into models
    f::Float64         # function-value of fn(x)
    v::Bool            # viability / feasibility
    c::Float64         # contravention / violation

    Agent(ND::Int) = new(Vector{Float64}(undef, ND), Inf, false, 0.0) # @code_warntype ✓
end

function Base.isequal(a1::Agent, a2::Agent) # @code_warntype ✓
    # a1, a2 are both feasible
    a1.v && a2.v && return a1.f == a2.f
    # a1, a2 are both infesasible
    !a1.v && !a2.v && return a1.c == a2.c
    return false
end

function Base.isless(a1::Agent, a2::Agent) # @code_warntype ✓
    # a1, a2 are both feasible
    a1.v && a2.v && return a1.f < a2.f
    # a1, a2 are both infesasible
    !a1.v && !a2.v && return a1.c < a2.c
    # if (a1, a2) = (feasible, infeasible), then a1 < a2 is true
    # if (a1, a2) = (infeasible, feasible), then a2 < a1 is false
    return a1.v
end

#### random initialization, @code_warntype ✓
function born!(x::VecIO, lb::NTuple{ND}, ub::NTuple{ND}) where ND
    @simd for i in eachindex(x)
        @inbounds x[i] = lb[i] + rand() * (ub[i] - lb[i])
    end
end

#### groups, @code_warntype ✓
function return_agents(ND::Int, NP::Int)
    agents = Vector{Agent}(undef, NP)
    @inbounds for i in eachindex(agents)
        agents[i] = Agent(ND)
    end
    return agents
end

#### subgroups, @code_warntype ✓
return_elites(agents::VecI{Agent}, NE::Int)          = view(agents, 1:NE)
return_throng(agents::VecI{Agent}, NE::Int, NP::Int) = view(agents, NE+1:NP)

#=
    Sine-Cosine Optimizer (https://doi.org/10.1016/j.knosys.2015.12.022)
    params:
    -------
    * Xb := buffer
    * Xn := n-th solution in the pool
    * Xr := referred solution
    * ss := step size
=#
function sco_move!(Xb::VecB, Xn::VecI, Xr::VecI, ss::Real) # @code_warntype ✓
    r = 2.0 * rand()
    s = sinpi(r)
    c = cospi(r)
    @simd for i in eachindex(Xb)
        @inbounds Xb[i] = Xn[i] + ss * abs(Xn[i] - Xr[i]) * ifelse(rand() < 0.5, s, c)
    end
end
#=
    Water-Cycle Algorithm Optimizer (https://doi.org/10.1016/j.compstruc.2012.07.010)
    params:
    -------
    * Xb    := buffer
    * Xbest := the best solution currently
=#
function wca_move!(Xb::VecB, Xbest::VecI) # @code_warntype ✓
    scaled_rand = randn() * 0.31622776601683794 # sqrt(0.1)
    @simd for i in eachindex(Xb)
        @inbounds Xb[i] = Xbest[i] + scaled_rand
    end
end

end # module
