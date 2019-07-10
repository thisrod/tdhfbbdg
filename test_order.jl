module OrderParameterTests

using Fields
using Test
using LinearAlgebra

using RK4IP

x, y = begin
	h = 0.1;  l = 15;  n = ceil(Int, l/h)
	R = XField([h, h], ones(n, n))
	grid(R)
end

@testset "lexp solves the 2D heat equation df/dt = ∇²f" begin

	f(t,x) = exp(-x^2/4t)/√t
	f(t,x,y) = f(t,x)*f(t,y)
	
	f₀ = apply_fields((x, y) -> f(0.5, x, y), x, y)
	τ = 0.1
	f₁ = lexp(f₀,τ)
	f₂ = apply_fields((x, y) -> f(0.5 + τ, x, y), x, y)
	
	@test f₁.vals ≈ f₂.vals
end

# Harmonic oscillator and ground state
V = x^2 + y^2
H(ψ) = -∇²(ψ) + V*ψ
φ₀ = exp(-(x^2+y^2)/2)/√π
@assert sum(abs2(φ₀)) ≈ 1

@testset "harmonic oscillator ground state" begin

	Δt = 5e-3

	D(ψ) = lexp(ψ, Δt/2)
	N(ψ) = -Δt*V*ψ
	
	# n-D ground state has energy n
	@test sum(conj(φ₀)*H(φ₀)) ≈ 2
	
	ψ = advance(φ₀,D,N);  ψ /= norm(ψ)
	@test ψ.vals ≈ φ₀.vals
	
	ψ₀ = φ₀*XField([h, h], 1 .+ 0.1*randn(n, n))
	ψ = advance(ψ₀,D,N);  ψ /= norm(ψ)
	@test norm(ψ-φ₀) < norm(ψ-ψ₀)
end

@testset "naive imaginary time" begin

	C = 10
	Δt = 1e-2
	
	L(ψ) = H(ψ) + C*abs2(ψ)*ψ
	D(ψ) = lexp(ψ, Δt/2)
	N(ψ) = -Δt*(V + C*abs2(ψ))*ψ
	
	ψ = deepcopy(φ₀)
	μs = [sum(conj(φ₀)*L(φ₀)).re,]
	rsdls = [norm(L(φ₀)-μs[1]*φ₀),]
	for i = 1:700
		ψ, μs
		ψ = advance(ψ, D, N)
		ψ /= norm(ψ)
		push!(μs, sum(conj(ψ)*L(ψ)).re)
		push!(rsdls, norm(L(ψ)-μs[end]*ψ))
	end
	
	@test abs(μs[end-1]-μs[end]) < 100*eps(μs[end])
	@test rsdls[end] > 0.01
	@test rsdls[end] > 0.9*rsdls[200]

end

@testset "harmonic oscillator ground state stable under Gauss-Seidel" begin
	x1, y1 = begin
		h = 0.1;  l = 15;  n = ceil(Int, l/h)
		R = XField([h, 1], ones(n, 1))
		grid(R)
	end
	
	V = x1^2
	φ₀ = π^(-1/4)*exp(-x1^2/2)
	@assert norm(φ₀) ≈ 1
	
	y₀ = φ₀.vals
	H = -lmat(V) + Diagonal(V.vals[:])
	y₁ = gauss_seidel_step(H, y₀, y₀)
	φ₁ = XField(φ₀.h, reshape(y₁, size(φ₀.vals)))
			
	@test φ₁.vals ≈ φ₀.vals
end

end # module

