using Plots, ComplexPhasePortrait

h = 0.05
N = 150
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = hypot.(y,x)
φ = atan.(y,x)

w = @. r*exp(1im*(φ+sin(φ)))/sqrt(1+r^2)

zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))

# concept: expand phase in least-squares Fourier series, compare constant to linear terms

function round_square(j, k, a)
    [tuple.(j+a:-1:j-a, k+a);
    tuple.(j-a,k+a-1:-1:k-a);
    tuple.(j-a+1:j+a,k-a);
    tuple.(j+1,k-a+1:k+a)]
end

function circn(w,j,k)
    wsq = [w[ix...] for ix = round_square(j,k,1)]
    wsq[2:end].*conj.(wsq[1:end-1]) .|> angle |> sum
end
