module OrderParameterTests

using Fields
using RK4IP
using Test

@testset "Operator exp(∇²) solves the heat equation" begin
	φ(x,t) = exp(-x^2/4t)/2/√(π*t)
	
	h = 0.1;  l = 15;  N = ceil(Int, l/h)
	R = XField([h, h], ones(N, N))
	x, y = grid(R)
	
	t = 0.5
	φ0 = apply_field(x -> φ(x,t), x) * apply_field(x -> φ(x,t), y)
	
	τ = 0.1
	φ1 = explap(φ0, τ)
	φ2 = apply_field(x -> φ(x,t+τ), x) * apply_field(x -> φ(x,t+τ), y)
	
	@test φ1.vals ≈ φ2.vals
end

@testset "harmonic oscillator ground state" begin
	V(x,y) = x^2 + y^2
	C = 0

	h = 0.1;  l = 15;  n = ceil(Int, l/h)
	Δt = 5e-3
	
	H(ψ) = -lap(ψ) + nlptl(ψ, V, C)
	D(ψ) = explap(ψ, Δt/2)
	N(ψ) = -Δt*nlptl(ψ, V, C)
	
	R = XField([h, h], ones(n, n))
	x, y = grid(R)
	ψ₀ = exp(-(x^2+y^2)/2)/√π
	
	# n-D ground state has energy n
	@test sum(conj(ψ₀)*H(ψ₀)) ≈ 2
	
	ψ = itime_step(ψ₀, D, N)
	ψ₁ = ψ/norm(ψ)
	
	@test ψ₁.vals ≈ ψ₀.vals
	
	φ₀ = ψ₀*XField([h, h], 1 .+ 0.1*randn(n, n))
	φ = itime_step(φ₀, D, N)
	φ₁ = φ/norm(φ)
	
	@test norm(φ₁-ψ₀) < norm(φ₁-φ₀)
end

end # module

using Fields
using RK4IP

	V(x,y) = x^2 + y^2
	C = 0.5

	h = 0.1;  l = 15;  n = ceil(Int, l/h)
	Δt = 1e-2
	
	H(ψ) = -lap(ψ) + nlptl(ψ, V, 0)
	L(ψ) = -lap(ψ) + nlptl(ψ, V, C)
	D(ψ) = explap(ψ, Δt/2)
	N(ψ) = -Δt*nlptl(ψ, V, C)
	
	R = XField([h, h], ones(n, n))
	x, y = grid(R)
	ψ₀ = exp(-(x^2+y^2)/2)/√π
	ψ = deepcopy(ψ₀)
	μs = zeros(101)
	μs[1] = sum(conj(ψ₀)*L(ψ₀)).re
	for i = 1:100
		global ψ = itime_step(ψ, D, N)
		ψ /= norm(ψ)
		global μs[i+1] = sum(conj(ψ)*L(ψ)).re
	end
	
#	@test ψ₁.vals ≈ ψ₀.vals
