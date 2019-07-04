module RK4IP

using Fields

export ∇², lexp, nlptl, advance

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
	
end