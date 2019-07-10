module RK4IP

using Fields
using LinearAlgebra

export ∇², lexp, nlptl, advance, gauss_seidel_step, adapt_relax

function ∇²(u)
	v = fft(u)
	kx, ky = grid(v)
	ifft(-(kx^2+ky^2) * v)
end

# exponential Laplacian.  Returns exp(τ∇²) u
function lexp(u, τ)
	v = fft(u)
	kx, ky = grid(v)
	ifft(exp(-τ*(kx^2+ky^2)) * v)
end

# Algorithm B.10 of Caradoc-Davies
function advance(ψ, D, N)
	ψ_I = D(ψ)
	k₁ = D(N(ψ))
	k₂ = N(ψ_I + k₁/2)
	k₃ = N(ψ_I + k₂/2)
	k₄ = N(D(ψ_I + k₃))
	@assert 0.5*norm(ψ) > norm(k₁) + norm(k₂) + norm(k₃) + norm(k₄) 
	D(ψ_I + (k₁ + 2(k₂+k₃))/6) + k₄/6
end

"""
    adapt_relax(step, x₀, Δt₀)

The function |x', R = step(x, Δt)| returns an updated value of |x| and a residual |R|.  The value x is iteratively updated, and the time step Δt is adjusted so that each step causes the maximum reduction in the residual.

TODO add termination conditions with defaults.
"""
function adapt_relax(step, x₀, Δt₀)

end

"""
    gauss_seidel_step(A, b, y₀)

Return y₁, the vector y₀ advanced by one step of the Gauss-Seigel algorithm.

Extra method.  A is a function.  A(y, i) returns A[i,:]⋅y, and A(i) returns 
"""
function gauss_seidel_step(A, b, y₀)
	y = copy(y₀)
	for i = eachindex(y)
		y[i] = 0
		y[i] = (b[i]-A[i,:]⋅y)/A[i,i]
	end
	y
end

end