using BandedMatrices
using Plots

h = 1
N = 7

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

# ∂² = (1/h^2).*op(Float64[1, -2, 1])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

P = []
for ix = [(4,4), (6,2), (1,1), (3,1)]
    u = zeros(7,7)
    u[ix...] = 1
    Lu = ∂²*u + u*∂²'
    push!(P, heatmap(Lu, aspect_ratio=1))
end

plot(P..., layout=@layout [a b; c d])