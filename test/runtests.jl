using GlobalOptimizer, Test

@testset "Solve Ackley Function" begin
    # Define an objective function (using Ackley function here as an example)
    function ackley(x::AbstractVector{T}) where T<:Real
        arg1 = 0.0
        arg2 = 0.0
        dims = length(x)
        @inbounds for i in eachindex(x)
            arg1 += abs2(x[i])
            arg2 += cospi(2.0 * x[i])
        end
        arg1  = 0.2 * sqrt(arg1 / dims)
        arg2 /= dims
        return -20.0 * exp(-arg1) - exp(arg2) + ℯ + 20.0
    end

    nd  = 15
    lb  = ntuple(i -> -32.0, nd) # lower bounds
    ub  = ntuple(i -> +32.0, nd) # upper bounds
    obj = minimizer(nd)          # create an object for the optimization
    minimize!(obj, ackley, lb, ub, avgtimes=3)
    @test obj.xsol ≈ zeros(Float64, nd) atol=1e-12
end
