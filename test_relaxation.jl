module RelaxationTest

include("RK4IP.jl")
using .RK4IP
using Test

@testset "adapt_relax with steady decay" begin

	target = 11
	function f(x, a)
		x₁ = x + a*(target-x)
		x₁, abs(target-x₁)
	end
	a₀ = 1
	
	@test adapt_relax(f, target, a₀, 100eps()) ≈ target
	@test adapt_relax(f, 0, a₀, 100eps()) ≈ target

end

end