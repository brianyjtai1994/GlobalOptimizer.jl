module GlobalOptimizer

############################
# Sorting                  #
############################

function swap!(v::VecIO, i::Int, j::Int) # @code_warntype ✓
    @inbounds temp = v[i]
    @inbounds v[i] = v[j]
    @inbounds v[j] = temp
    return nothing
end

biinsert(arr::VecI, val::T) where T = biinsert(arr, val, 1, length(arr)) # @code_warntype ✓
function biinsert(arr::VecI, val::T, lx::Int, rx::Int) where T           # @code_warntype ✓
    lx ≥ rx && return lx
    ub = rx # upper bound
    while lx < rx
        mx = (lx + rx) >> 1                                    # midpoint (binary search)
        @inbounds isless(val, arr[mx]) ? rx = mx : lx = mx + 1 # arr[mx].f == val in this case
    end
    @inbounds lx == ub && !isless(val, arr[lx]) && (lx += 1)   # lx = upper bound && arr[lx] ≤ val
    return lx
end

binsort!(arr::VecI) = binsort!(arr, 1, length(arr)) # @code_warntype ✓
function binsort!(arr::VecI, lx::Int, rx::Int)      # @code_warntype ✓
    for ix in lx+1:rx
        @inbounds val = arr[ix]
        jx = ix
        lc = biinsert(arr, val, lx, ix) # location
        while jx > lc
            swap!(arr, jx, jx - 1)
            jx -= 1
        end
    end
end

############################
# Agent                    #
############################

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

############################
# Dealing with constraints #
############################

struct BoxBound a::Float64; b::Float64; i::Int end # a * x[i] + b

resolve_lb(lb::Real) = iszero(lb) ? (-1.0, 0.0) : (-abs(inv(lb)),  1.0 * sign(lb)) # @code_warntype ✓
resolve_ub(ub::Real) = iszero(ub) ? ( 1.0, 0.0) : ( abs(inv(ub)), -1.0 * sign(ub)) # @code_warntype ✓

function boxbounds(lb::NTuple{ND}, ub::NTuple{ND}) where ND
    if @generated
        a = Vector{Expr}(undef, 2*ND)
        @inbounds for i in 1:ND
            a[i]    = :(BoxBound(resolve_lb(lb[$i])..., $i))
            a[i+ND] = :(BoxBound(resolve_ub(ub[$i])..., $i))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, a...))
        end
    else
        return ntuple(i -> i > ND ? BoxBound(resolve_ub(ub[i - ND])..., i - ND) : BoxBound(resolve_lb(lb[i])..., i), 2*ND)
    end
end

eval_violation(x::VecI, bb::BoxBound) = max(0.0, bb.a * x[bb.i] + bb.b) # @code_warntype ✓
function eval_violation(x::VecI, cons::NTuple{NB,BoxBound}) where NB    # @code_warntype ✓
    if @generated
        a = Vector{Expr}(undef, NB)
        @inbounds for i in eachindex(a)
            a[i] = :(eval_violation(x, cons[$i]))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:call, :+, a...))
        end
    else
        ret = 0.0
        @inbounds for i in eachindex(cons)
            ret += eval_violation(x, cons[i])
        end
        return ret
    end
end

# check feasibility of agent in throng, @code_warntype ✓
function check!(xnew::VecI, agents::VecIO{Agent}, elites::VecIO{Agent}, throng::VecIO{Agent}, edx::Int, tdx::Int, fn::Function, cons::NTuple)
    violation = eval_violation(xnew, cons)
    violation > 0.0 && return @inbounds check!(xnew, violation, throng[tdx]) # x[new] is infeasible
    return check!(xnew, fcall(fn, xnew), agents, elites, throng, edx, tdx)   # x[new] is feasible
end

# Matchup for a feasible x[new] agent in throng, @code_warntype ✓
function check!(xnew::VecI, fnew::Real, agents::VecIO{Agent}, elites::VecIO{Agent}, throng::VecIO{Agent}, edx::Int, tdx::Int)
    @inbounds xold = throng[tdx]
    # x[old] is infeasible
    if !xold.v
        xold.f = fnew
        xold.v = true
        xold.c = 0.0
        copy!(xold.x, xnew)
        return nothing
    end
    # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
    if @inbounds !(elites[edx].f < fnew) 
        xold.f = fnew
        copy!(xold.x, xnew)
        swap!(agents, edx, length(elites) + tdx)
        return nothing
    end
    # x[old], x[new] are feasible
    if !(xold.f < fnew) 
        xold.f = fnew
        copy!(xold.x, xnew)
        return nothing
    end
end

# check feasibility of agent in elites, @code_warntype ✓
function check!(xnew::VecI, elites::VecIO{Agent}, edx::Int, fn::Function, cons::NTuple)
    violation = eval_violation(xnew, cons)
    violation > 0.0 && return @inbounds check!(xnew, violation, elites[edx]) # x[new] is infeasible
    return check!(xnew, fcall(fn, xnew), elites, edx)                        # x[new] is feasible
end

# Matchup for a feasible x[new] trial in elites, @code_warntype ✓
function check!(xnew::VecI, fnew::Real, elites::VecIO{Agent}, edx::Int)
    @inbounds elite = elites[edx]
    # x[old] is infeasible
    if !elite.v
        elite.f = fnew
        elite.v = true
        elite.c = 0.0
        copy!(elite.x, xnew)
        return nothing
    end
    # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
    if @inbounds !(elites[1].f < fnew)
        elite.f = fnew
        copy!(elite.x, xnew)
        swap!(elites, 1, edx)
        return nothing
    end
    # x[old], x[new] are feasible
    if !(elite.f < fnew) 
        elite.f = fnew
        copy!(elite.x, xnew)
        return nothing
    end
end

# Matchup for an infeasible x[new] trial, here "fnew = violation", @code_warntype ✓
function check!(xnew::VecI, violation::Real, agent::Agent)
    # x[old], x[new] are infeasible, compare violation
    # There is no `else` condition, if x[old] is feasible, then a matchup is unnecessary.
    if !agent.v && !(agent.c < violation)
        agent.c = violation
        copy!(agent.x, xnew)
    end
    return nothing
end

end # module
