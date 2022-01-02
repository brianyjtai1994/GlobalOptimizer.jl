module GlobalOptimizer

const VecI  = AbstractVector # Input  Vector
const VecO  = AbstractVector # Output Vector
const VecB  = AbstractVector # Buffer Vector
const VecIO = AbstractVector # In/Out Vector

export minimize!, minimizer

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

    Agent(nd::Int) = new(Vector{Float64}(undef, nd), Inf, false, 0.0) # @code_warntype ✓
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
function return_agents(nd::Int, np::Int)
    agents = Vector{Agent}(undef, np)
    @inbounds for i in eachindex(agents)
        agents[i] = Agent(nd)
    end
    return agents
end

#### subgroups, @code_warntype ✓
return_elites(agents::VecI{Agent}, ne::Int)          = view(agents, 1:ne)
return_throng(agents::VecI{Agent}, ne::Int, np::Int) = view(agents, ne+1:np)

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

fcall(f::Function, x::VecI) = f(x)

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

############################
# Minimization             #
############################

## Generic logistic function
logistic(x::Real, x0::Real, a::Real, k::Real, c::Real) = a / (1.0 + exp(k * (x0 - x))) + c

## Perform a single step of Wolford algorithm
function welford_step(μ::Real, s::Real, v::Real, c::Real)
    isone(c) && return v, zero(v)
    s = s * (c - 1)
    m = μ + (v - μ) / c
    s = s + (v - μ) * (v - m)
    μ = m
    return μ, s / (c - 1)
end

function nrm2(x::VecI{Tx}, y::VecI{Ty}, b::VecB{Tb}) where {Tx<:Real,Ty<:Real,Tb<:Real} # @code_warntype ✓
    @simd for i in eachindex(b)
        @inbounds b[i] = abs2(x[i] - y[i])
    end
    return sqrt(sum(b))
end

struct GlobalMinimizer
    xsol::Vector{Float64}
    xerr::Vector{Float64}
    buff::Vector{Float64}
    fork::Vector{Int64}
    pool::Vector{Agent}
    np::Int
    ne::Int

    function GlobalMinimizer(nd::Int, np::Int, ne::Int)
        xsol = Vector{Float64}(undef, nd)
        xerr = Vector{Float64}(undef, nd)
        buff = Vector{Float64}(undef, nd)
        fork = Vector{Int}(undef, ne)
        pool = return_agents(nd, np)
        return new(xsol, xerr, buff, fork, pool, np, ne)
    end
end

# @code_warntype ✓
function inits!(agents::VecIO{Agent}, lb::NTuple, ub::NTuple)
    for agent in agents
        born!(agent.x, lb, ub)
    end
end

# @code_warntype ✓
function inits!(agents::VecIO{Agent}, f::Function, cons::NTuple)
    fmax = -Inf
    for agent in agents
        violation = eval_violation(agent.x, cons)
        violation > 0.0 && (agent.c = violation; continue) # agent is infeasible
        agent.v = true
        agent.f = fcall(f, agent.x)
        fmax    = max(fmax, agent.f)
    end
    for agent in agents
        !agent.v && (agent.f = agent.c + fmax) # agent is infeasible
    end
end

# @code_warntype ✓
function group!(fork::VecIO{Int}, agents::VecI{Agent}, ne::Int, nc::Int)
    diversity = 0.0
    @inbounds for i in eachindex(fork)
        diversity += agents[ne + 1].f - agents[i].f
    end
    if iszero(diversity) || isnan(diversity)
        fill!(fork, 1)
    else
        @inbounds for i in eachindex(fork)
            fork[i] = max(1, round(Int, nc * (agents[ne + 1].f - agents[i].f) / diversity))
        end
    end
    res = nc - sum(fork) # residue
    idx = 2
    while res > 0
        @inbounds fork[idx] += 1; res -= 1
        idx < ne ? idx += 1 : idx = 2
    end
    while res < 0
        @inbounds fork[idx] = max(1, fork[idx] - 1); res += 1
        idx < ne ? idx += 1 : idx = 2
    end
end

# @code_warntype ✓
function minimize!(o::GlobalMinimizer, fn::Function, lb::NTuple{ND,T}, ub::NTuple{ND,T}, itmax::Int, dmax::Real, avgtimes::Int) where {ND,T<:Real}
    np = o.np
    ne = o.ne
    nc = np - ne

    cons = boxbounds(lb, ub)
    xsol = o.xsol
    xerr = o.xerr
    buff = o.buff
    fork = o.fork

    agents = o.pool
    elites = return_elites(agents, ne)
    throng = return_throng(agents, ne, np)

    generation = 0
    while generation < avgtimes
        generation += 1
        itcount     = 0

        inits!(agents, lb, ub)
        inits!(agents, fn, cons)
        binsort!(agents)

        @inbounds while itcount < itmax
            itcount += 1
            ss = logistic(itcount, 0.5 * itmax, -0.618, 20.0 / itmax, 2.0)
            group!(fork, agents, ne, nc)

            #### Moves: throng → elites, elites → the-best
            rx = 1
            fx = fork[rx]
            # move agents (in throng) → elites
            for ix in eachindex(throng)
                sco_move!(buff, elites[rx].x, throng[ix].x, ss)
                check!(buff, agents, elites, throng, rx, ix, fn, cons)
                fx -= 1
                iszero(fx) && rx < ne && (rx += 1; fx = fork[rx])
            end
            # move agents (in elites) and find the best one
            for rx in 2:ne
                sco_move!(buff, elites[1].x, elites[rx].x, ss)
                check!(buff, elites, rx, fn, cons)
            end

            #### Random searching process
            for ix in 1:fork[1]
                if !(dmax < nrm2(agents[1].x, throng[ix].x, buff))
                    wca_move!(buff, agents[1].x)
                    check!(buff, agents, elites, throng, 1, ix, fn, cons)
                end
            end
            for rx in 2:ne
                if !(dmax < nrm2(agents[1].x, elites[rx].x, buff)) || !(0.1 < rand())
                    born!(buff, lb, ub)
                    check!(buff, elites, rx, fn, cons)
                end
            end

            #### Update the function-value of infeasible candidates
            fmax = -Inf
            for agent in agents
                agent.v && (fmax = max(fmax, agent.f))
            end
            for agent in agents
                !agent.v && (agent.f = agent.c + fmax)
            end

            binsort!(agents)
            dmax -= dmax / itmax
        end

        @inbounds xnew = agents[1].x
        @inbounds for i in eachindex(xsol)
            xsol[i], xerr[i] = welford_step(xsol[i], xerr[i], xnew[i], generation)
        end
    end
end

############################
# Interface                #
############################

"""
    minimizer(nd::Int; np::Int=35*nd, ne::Int=nd+1)
An interface function to create/initialize an object for the minimization.
Arguments:
---
- `nd`:     Dimension of parameters to be minimized.
- `np`:     Desired population size (*optional*).
- `ne`:     Desired size of elites (*optional*).
"""
minimizer(nd::Int; np::Int=35*nd, ne::Int=nd+1) = GlobalMinimizer(nd, np, ne)

"""
    minimize!(o::GlobalMinimizer, fn::Function, lb::NTuple{ND}, ub::NTuple{ND}; itmax::Int=210*ND, dmax::Real=1e-7, avgtimes::Int=1)
An interface function to proceed the "global" minimization.
Arguments:
---
- `o`:        An object for the minimization created by `minimizer(...; args...)`.
- `fn`:       Objective function to be minimized. 
              `fn` should be callable with only one argument of `fn(x::Vector)`. 
              If you have any additional arguments need to pass into it, 
              dispatch the function by `fcall(fn, x; kwargs...) = fn(x; kwargs...)`
- `lb`:       Lower bounds of minimization which are forced to be feasible.
- `ub`:       Upper bounds of minimization which are forced to be feasible.
- `itmax`:    Maximum of minimizing iteration (*optional*).
- `dmax`:     An Euclidean distance acts as a criterion to
              prevent the population falling into local minimal (*optional*).
- `avgtimes`: Number of average times of the whole minimization process (*optional*).
"""
minimize!(o::GlobalMinimizer, fn::Function, lb::NTuple{ND}, ub::NTuple{ND}; itmax::Int=210*ND, dmax::Real=1e-7, avgtimes::Int=1) where ND = minimize!(o, fn, lb, ub, itmax, dmax, avgtimes)

end # module
